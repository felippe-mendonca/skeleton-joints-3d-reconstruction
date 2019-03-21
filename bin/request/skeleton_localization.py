import json
from argparse import ArgumentParser
from sys import exit
from os import makedirs, walk
from os.path import join, dirname, exists, basename
from shutil import rmtree
import numpy as np
import pandas as pd

from is_wire.core import Channel, Logger
from is_msgs.image_pb2 import ObjectAnnotations
from src.utils.proto.group_request_pb2 import MultipleObjectAnnotations
from src.utils.is_wire import RequestManager
from src.utils.is_msgs import data_frame_to_object_annotations, object_annotations_to_np
from src.panoptic_dataset.utils import is_valid_model, make_df_columns, RESOLUTION

log = Logger(name='SkeletonLocalization')


def main(sequence_folder, output_folder, pose_model, cameras, broker_uri, min_requests,
         max_requests, timeout_ms):

    info_file_path = join(sequence_folder, 'info.json')
    if not exists(info_file_path):
        log.critical("'{}' file doesn't exist.", info_file_path)

    with open(info_file_path, 'r') as f:
        sequence_info = json.load(f)

    try:
        is_valid_model(pose_model)
    except Exception as ex:
        log.critical(str(ex))

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

    sequence_name = basename(dirname(sequence_folder))
    received_data = []
    while True:

        while request_manager.can_request() and len(sample_ids) > 0:
            sample_id = sample_ids.pop(0)
            request = make_request(sample_id)
            request_manager.request(
                content=request,
                topic="SkeletonsGrouper.Localize",
                timeout_ms=timeout_ms,
                metadata=sample_id)
            log.info("[{}] [{:>3s}] {}", sequence_name, ">>", sample_id)

        received_msgs = request_manager.consume_ready(timeout=1.0)

        for msg, received_sample_id in received_msgs:
            localizations = msg.unpack(ObjectAnnotations)
            localizations_array = object_annotations_to_np(
                annotations_pb=localizations,
                model=pose_model,
                has_z=True,
                add_person_id=True,
                sample_id=received_sample_id)
            received_data.append(localizations_array)

            log.info("[{}] [{:<3s}] {}", sequence_name, "<<", received_sample_id)

        if request_manager.all_received() and len(sample_ids) == 0:
            log.info("All received.")
            received_data = np.vstack(received_data)
            df = pd.DataFrame(data=received_data, columns=make_df_columns(pose_model))
            df.sort_values(by=['sample_id', 'person_id'], axis='rows', inplace=True)

            output_folder_path = join(output_folder, sequence_name, pose_model)
            if exists(output_folder_path):
                rmtree(output_folder_path)
            makedirs(output_folder_path)
            output_file_path = join(output_folder_path, 'data.csv')

            log.info("Saving results on {}", output_file_path)
            df.to_csv(path_or_buf=output_file_path, header=True, index=False)

            break


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    parser.add_argument(
        '--config-file',
        type=str,
        required=False,
        help="""JSON configuration file with parameters. Arguments passed on
        command line will overwrite configuration file.""")
    parser.add_argument('--help', '-h', action='store_true')

    args, unknown = parser.parse_known_args()

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
        '--output-folder',
        type=str,
        required=True,
        help="""Path to folder to save a data.csv file with results.
        A folder with the sequence name and another inside that with the 
        pose model will be created to save this file.""")
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

    parsed_from_file = []

    if args.help:
        parser.print_help()
        exit(-1)
    elif args.config_file is not None:
        with open(args.config_file, 'r') as f:
            args_from_file = json.load(f)
        for arg_key in parser._option_string_actions.keys():
            if not arg_key.startswith('--'):
                continue
            no_dash_key = arg_key.split('--')[-1]
            if no_dash_key in args_from_file:
                arg_values = args_from_file[no_dash_key]
                if type(arg_values) == list:
                    arg_values = list(map(str, arg_values))
                else:
                    arg_values = [arg_values]
                parsed_from_file.extend([arg_key] + arg_values)

    args = parser.parse_args(parsed_from_file + unknown)

    main(
        sequence_folder=args.sequence_folder,
        output_folder=args.output_folder,
        pose_model=args.pose_model,
        cameras=args.cameras,
        broker_uri=args.broker_uri,
        min_requests=args.min_requests,
        max_requests=args.max_requests,
        timeout_ms=args.timeout_ms)
