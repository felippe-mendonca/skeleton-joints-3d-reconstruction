import json
from argparse import ArgumentParser
from os import makedirs, walk
from os.path import join, dirname, exists
import pandas as pd

from is_wire.core import Channel, Logger
from is_msgs.image_pb2 import ObjectAnnotations
from src.utils.proto.group_request_pb2 import MultipleObjectAnnotations
from src.utils.is_wire import RequestManager
from src.utils.is_msgs import data_frame_to_object_annotations
from src.panoptic_dataset.utils import AVAILABLE_MODELS, RESOLUTION

log = Logger(name='SkeletonLocalization')


def main(sequence_folder, pose_model, cameras, broker_uri, min_requests, max_requests, timeout_ms):
    info_file_path = join(sequence_folder, 'info.json')
    if not exists(info_file_path):
        log.critical("'{}' file doesn't exist.", info_file_path)

    with open(info_file_path, 'r') as f:
        sequence_info = json.load(f)

    if pose_model not in AVAILABLE_MODELS:
        log.critical("Invalid pose model '{}'.", pose_model)

    annotations_folder_path = join(sequence_folder, '2d_annotations', pose_model)
    _, _, annotations_files_available = next(walk(annotations_folder_path))

    available_cameras = list(map(lambda x: int(x.strip('.csv')), annotations_files_available))
    not_available_cameras = set(cameras).difference(available_cameras)
    if len(not_available_cameras) > 0:
        nav_cam_str = ', '.join(map(str, sorted(not_available_cameras)))
        av_cam_str = ', '.join(map(str, sorted(available_cameras)))
        log.critical(
            "For sequence {}, model {}, camera(s) {} are not available. Only {} are present. Exiting.",
            sequence_folder, pose_model, nav_cam_str, av_cam_str)

    annotations_data = {}
    for camera in cameras:
        annotation_file_path = join(annotations_folder_path, '{}.csv'.format(camera))
        annotations_data[camera] = pd.read_csv(annotation_file_path)

    def make_request(sample_id):
        m_obj_annotations = MultipleObjectAnnotations()
        for camera in cameras:
            annotations = annotations_data[camera]
            sample_annotations = annotations[annotations['sample_id'] == sample_id]
            obj_annotations = data_frame_to_object_annotations(
                annotations=sample_annotations,
                model=pose_model,
                frame_id=camera,
                resolution=RESOLUTION)
            m_obj_annotations.list.add().CopyFrom(obj_annotations)

        return m_obj_annotations

    sample_ids = list(range(sequence_info['begin'], sequence_info['end'] + 1))

    channel = Channel(broker_uri)
    request_manager = RequestManager(
        channel=channel, max_requests=max_requests, min_requests=min_requests)

    while True:

        while request_manager.can_request() and len(sample_ids) > 0:
            sample_id = sample_ids.pop()
            request = make_request(sample_id)
            request_manager.request(
                content=request,
                topic="SkeletonsGrouper.Localize",
                timeout_ms=timeout_ms,
                metadata=sample_id)

        received_msgs = request_manager.consume_ready(timeout=1.0)

        for msg, received_sample_id in received_msgs:
            localizations = msg.unpack(ObjectAnnotations)

        if request_manager.all_received() and len(sample_ids) == 0:
            log.info("All received. Exiting.")
            break


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '--sequence-folder',
        type=str,
        required=True,
        help="""Path to folder containing a sequence from CMU Panoptic dataset.
        This folder must have a '2d_annotations' folder containing a folder
        named with the pose model, i.e., 'joints15' or 'joints19'. Besides, 
        the sequence folder might have a 'info.json' file that is generated
        by running the 'convert_3d_annotations' script.""")
    parser.add_argument(
        '--pose-model',
        type=str,
        required=False,
        default='joints19',
        help="""You can specify what model to process, can be either 'joints15' 
        or 'joints19'.""")
    parser.add_argument(
        '--cameras',
        type=int,
        required=True,
        nargs='+',
        help="""Cameras need to be specified with their ids. If a specified 
        camera doesn't have the 2D annotations file related to itself, the
        program will terminate.""")
    parser.add_argument(
        '--broker-uri',
        type=str,
        required=False,
        default='amqp://localhost:5672',
        help="""RabbitMQ Broker URI to connect and send request to SkeletonGrouper.Localize.""")
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
        default=1000,
        help="""ResquestManager parameter. Amount of time to a sent message receive a 
        response. In case of reach this deadline, RequestManager will retry indefinitely.""")

    args = parser.parse_args()
    main(
        sequence_folder=args.sequence_folder,
        pose_model=args.pose_model,
        cameras=args.cameras,
        broker_uri=args.broker_uri,
        min_requests=args.min_requests,
        max_requests=args.max_requests,
        timeout_ms=args.timeout_ms)
