import json
from sys import exit
from os import makedirs, walk
from os.path import join, dirname, exists, basename
from shutil import rmtree
from urllib.parse import urlparse
import numpy as np
import pandas as pd

from is_wire.core import Channel, Logger
from is_wire.core import ZipkinExporter, BackgroundThreadTransport
from is_msgs.image_pb2 import ObjectAnnotations
from src.utils.arparse import ArgumentParserFile
from src.utils.is_wire import RequestManager
from src.utils.video import VideoIterator
from src.utils.image import get_pb_image

from src.panoptic_dataset.utils import is_video_file, get_camera_id, make_df_columns
from src.utils.is_msgs import object_annotations_to_np

log = Logger(name='SkeletonDetection')


def main(sequence_folder, output_folder, info_folder, pose_model, broker_uri, zipkin_uri,
         min_requests, max_requests, timeout_ms):

    _, _, video_files = next(walk(sequence_folder))
    video_files = list(filter(is_video_file, video_files))

    sequence_name = basename(dirname(sequence_folder + '/'))
    info_file_path = join(info_folder, sequence_name, 'info.json')

    if not exists(info_file_path):
        begin_id, end_id = 0, -1
        log.warn("Can't find info file on '{}'", info_file_path)
    else:
        with open(info_file_path) as f:
            sequence_info = json.load(f)
        begin_id, end_id = sequence_info['begin'], sequence_info['end']
    
    output_folder_path = join(output_folder, sequence_name, '2d_annotations', pose_model)
    if exists(output_folder_path):
        rmtree(output_folder_path)
    makedirs(output_folder_path)

    channel = Channel(broker_uri)
    zipkin_exporter = None

    if zipkin_uri is not None:
        zipkin_uri = urlparse(zipkin_uri)
        zipkin_exporter = ZipkinExporter(
            service_name="RequestSkeletonsDetection",
            host_name=zipkin_uri.hostname,
            port=zipkin_uri.port,
            transport=BackgroundThreadTransport(max_batch_size=100),
        )

    request_manager = RequestManager(
        channel=channel,
        zipkin_exporter=zipkin_exporter,
        max_requests=max_requests,
        min_requests=min_requests)

    for video_file in video_files:
        video_file_path = join(sequence_folder, video_file)

        video_iterator = VideoIterator(video_file_path)
        it_range = range(begin_id, end_id + 1)
        data_iterator = zip(it_range, video_iterator.in_range(it_range))

        received_data = []
        camera_id = get_camera_id(video_file)
        end_of_data = False

        while True:

            while request_manager.can_request() and not end_of_data:
                try:
                    sample_id, frame = next(data_iterator)
                except StopIteration:
                    end_of_data = True
                    break

                request = get_pb_image(frame)
                metadata = {
                    "sample_id": sample_id,
                    "camera_id": camera_id,
                    "sequence": sequence_name,
                }
                request_manager.request(
                    content=request,
                    topic="SkeletonsDetector.Detect",
                    timeout_ms=timeout_ms,
                    metadata=metadata)

                log.info("[{}][{}][{:>3s}] {}", sequence_name, camera_id, ">>", sample_id)

            received_msgs = request_manager.consume_ready(timeout=1.0)

            for msg, received_metadata in received_msgs:
                localizations = msg.unpack(ObjectAnnotations)
                received_sample_id = received_metadata['sample_id']

                localizations_array = object_annotations_to_np(
                    annotations_pb=localizations,
                    model=pose_model,
                    has_z=False,
                    add_person_id=True,
                    sample_id=received_sample_id)
                received_data.append(localizations_array)

                log.info("[{}][{}][{:<3s}] {}", sequence_name, camera_id, "<<", received_sample_id)

            if request_manager.all_received() and end_of_data:
                log.info("All received.")
                received_data = np.vstack(received_data)
                columns = make_df_columns(pose_model, has_z=False)
                df = pd.DataFrame(data=received_data, columns=columns)
                df.sort_values(by=['sample_id', 'person_id'], axis='rows', inplace=True)

                output_file_path = join(output_folder_path, '{}.csv'.format(camera_id))
                log.info("Saving results on {}", output_file_path)
                df.to_csv(path_or_buf=output_file_path, header=True, index=False)

                break


if __name__ == '__main__':
    parser = ArgumentParserFile(parse_from_file=True)
    parser.add_argument(
        '--sequence-folder',
        type=str,
        required=True,
        help="""Path to folder containing a sequence from CMU Panoptic dataset.
        This folder must have MP4 files named with the pattern 'hd_00_{camera_id:02d}.mp4'.
        All videos inside that folder will be processed.""")
    parser.add_argument(
        '--output-folder',
        type=str,
        required=True,
        help="""Path to folder to save a CSV file for each camera containing all detections.""")
    parser.add_argument(
        '--info-folder',
        type=str,
        required=True,
        help="""Path to folder, containing a folder inside with the sequence name, 
        and inside that a 'info.json' with begin and end ids of the sequence. 
        If no specified, all frames will pre processed.""")
    parser.add_argument(
        '--pose-model',
        type=str,
        required=False,
        default='joints19',
        help="""You can specify what model, can be either 'joints15' 
        or 'joints19'. This will be used to save the output data correctly.""")
    parser.add_argument(
        '--broker-uri',
        type=str,
        required=False,
        default='amqp://localhost:5672',
        help="""RabbitMQ Broker URI to connect and send request to SkeletonGrouper.Localize.""")
    parser.add_argument(
        '--zipkin-uri',
        type=str,
        required=False,
        help="""Zipkin URI to export tracings from requests.""")
    parser.add_argument(
        '--min-requests',
        type=int,
        required=False,
        default=0,
        help="""ResquestManager parameter. Number of minimum requests to have on queue 
        waiting for a response. If not specified will be set to zero, which means that 
        request will be done only after receive all previous requests.""")
    parser.add_argument(
        '--max-requests',
        type=int,
        required=False,
        default=100,
        help="""ResquestManager parameter. Number of maximum requests to have on queue 
        waiting for a response.""")
    parser.add_argument(
        '--timeout-ms',
        type=int,
        required=False,
        default=5000,
        help="""ResquestManager parameter. Amount of time to a sent message receive a 
        response. In case of reach this deadline, RequestManager will retry indefinitely.""")

    args = parser.parse_args()

    main(
        sequence_folder=args.sequence_folder,
        output_folder=args.output_folder,
        info_folder=args.info_folder,
        pose_model=args.pose_model,
        broker_uri=args.broker_uri,
        zipkin_uri=args.zipkin_uri,
        min_requests=args.min_requests,
        max_requests=args.max_requests,
        timeout_ms=args.timeout_ms)
