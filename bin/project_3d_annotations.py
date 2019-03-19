import json
from argparse import ArgumentParser
from os import makedirs, walk
from os.path import join, dirname, exists
from shutil import rmtree
import numpy as np
import pandas as pd

from src.utils.numpy import to_np
from src.utils.cv import to_camera, validate_resolution
from src.utils.is_msgs import load_camera_calibration
from src.panoptic_dataset.utils import is_sequence_folder
from src.panoptic_dataset.utils import AVAILABLE_MODELS
from src.utils.logger import Logger

BASEDIR = join(dirname(__file__))

log = Logger(name='Project3DAnnotations')


def main(dataset_folder, pose_model, cameras):

    if not exists(dataset_folder):
        raise Exception("'{}' folder doesn't exist.".format(dataset_folder))

    _, sequence_folders, _ = next(walk(dataset_folder))
    sequence_folders = list(filter(is_sequence_folder, sequence_folders))

    for s_folder in sequence_folders:
        calibs_folder_path = join(dataset_folder, s_folder, 'calibrations')
        _, _, calib_files = next(walk(calibs_folder_path))

        if cameras is not None:
            available_cameras = list(map(lambda x: int(x.strip('.json')), calib_files))
            not_available_cameras = set(cameras).difference(available_cameras)
            if len(not_available_cameras) > 0:
                log.error("For sequence {}, camera(s) {} are not available. Skipping.", s_folder,
                          ', '.join(map(str, not_available_cameras)))
                continue

            calib_files = ['{}.json'.format(camera) for camera in cameras]

        calibs = {}
        for calib_file in calib_files:
            calib_file_path = join(calibs_folder_path, calib_file)
            calib = load_camera_calibration(calib_file_path)
            calibs[int(calib.id)] = calib

        annotations_folder_path = join(dataset_folder, s_folder, '3d_annotations')
        _, available_pose_models, _ = next(walk(annotations_folder_path))

        if len(available_pose_models) == 0:
            log.warn("For sequence {}, there is no annotation available. Skipping.", s_folder)
            continue

        if pose_model is not None:
            pose_model_warn = "For sequence {}, required pose model '{}' isn't available"
            pose_model_warn = pose_model_warn.format(s_folder, pose_model)
            if pose_model not in AVAILABLE_MODELS:
                log.warn("{}. Procceding with availables models on folder.", pose_model_warn)
            elif pose_model not in available_pose_models:
                log.warn("{} on annotations folder. Skipping.", pose_model_warn)
                continue
            else:
                available_pose_models = [pose_model]

        image_annotations_base_folder_path = join(dataset_folder, s_folder, '2d_annotations')

        for p_model in available_pose_models:
            poses_file_path = join(annotations_folder_path, p_model, 'data.csv')
            df = pd.read_csv(poses_file_path)
            # includes samples id and person id
            df_ids = df[['sample_id', 'person_id']].values
            pose_data = df.drop(['sample_id', 'person_id'], axis=1).values
            # Transform to a matrix with rows containing respectively x, y, z and confidence
            pose_data = pose_data.reshape(-1, 4).T
            joints_world_coordinate = pose_data[0:3, :]
            joints_confidences = pose_data[3, :]
            # not annotated joints are represented with all coordinates equals zero
            joints_not_annotated = (joints_world_coordinate == 0.0).all(axis=0)

            image_annotations_folder_path = join(image_annotations_base_folder_path, p_model)
            if exists(image_annotations_folder_path):
                rmtree(image_annotations_folder_path)
            makedirs(image_annotations_folder_path)

            for camera, calib in calibs.items():
                K = to_np(calib.intrinsic)
                RT = to_np(calib.extrinsic[0].tf)
                d = to_np(calib.distortion)
                w, h = calib.resolution.width, calib.resolution.height

                joints_camera_coordinate = to_camera(joints_world_coordinate, K, RT, d)
                joints_valid_resolution = validate_resolution(joints_camera_coordinate, w, h)
                invalid_joints = np.logical_or(joints_not_annotated, ~joints_valid_resolution)
                joints_camera_coordinate[:, invalid_joints] = 0.0

                image_pose_data = np.vstack([joints_camera_coordinate, joints_confidences])
                image_pose_data = image_pose_data.T.ravel().reshape(df.shape[0], -1)

                df_columns = list(filter(lambda x: 'z' not in x, list(df.columns)))
                data = np.hstack([df_ids, image_pose_data])
                df_image = pd.DataFrame(data=data, columns=df_columns)

                # drop invalid annotations, i.e., all x and y coordinates equals to zero
                invalid_rows = (df_image.filter(regex='^j[0-9]+[xy]$') == 0.0).all(axis=1)
                invalid_indexes = df_image[invalid_rows].index
                df_image.drop(invalid_indexes, inplace=True)

                log.info("Writing data from sequence {}, pose model {}, camera {}", s_folder,
                         p_model, camera)

                annotations_file_path = join(image_annotations_folder_path,
                                             '{}.csv'.format(camera))
                df_image.to_csv(path_or_buf=annotations_file_path, header=True, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '--dataset-folder',
        type=str,
        required=True,
        help="""Path to folder containing processed CMU Panoptic folders,
        with folders for each sequence, containing a '3d_annotations' folder
        inside. Both models, 'joints15' and 'joints19' will be processed if 
        none of them were specified. Projections for each specified camera 
        will be saved on a '2d_annotations' folder at the same level of 
        '3d_annotations'. Calibrations files must be inside of a 
        'calibrations' folder of each sequence .""")
    parser.add_argument(
        '--pose-model',
        type=str,
        required=False,
        help="""You can specify what model to process, can be either 'joints15' 
        or 'joints19'. If none of them were specified, all available on 
        '3d_annotations' folder will be processed.""")
    parser.add_argument(
        '--cameras',
        type=int,
        required=False,
        nargs='+',
        help="""Cameras can be specified with their ids to only compute its 
        projections. If no camera was specified, projections will be computed 
        to all cameras with available calibrations of sequence folder.""")

    args = parser.parse_args()
    main(args.dataset_folder, args.pose_model, args.cameras)