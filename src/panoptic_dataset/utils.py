import re
import json
from os.path import join, exists, dirname
import numpy as np

from is_msgs.camera_pb2 import CameraCalibration
from src.utils.numpy import to_tensor

SEQUENCE_PATTERN = re.compile(r'^[0-9]{6}_[a-zA-Z]+[0-9]{1}$')
POSE_FOLDER_PATTERN = re.compile(r'^hdPose3d_stage1(.*)')
SAMPLE_ID = re.compile(r'^body3DScene_([0-9]+).json$')
AVAILABLE_MODELS = ['joints15', 'joints19']
RESOLUTION = (1920, 1080)
VIDEO_FILE_PATTERN = re.compile(r'^hd_00_([0-9]{2}).mp4')


def is_valid_model(model):
    if model not in AVAILABLE_MODELS:
        raise Exception("Invalid Model: {}. Can be either 'joints15' or 'joints19'", model)


def is_sequence_folder(s):
    return SEQUENCE_PATTERN.match(s) is not None


def is_pose_folder(folder):
    return POSE_FOLDER_PATTERN.match(folder) is not None


def get_joints_key(folder):
    pose_model = POSE_FOLDER_PATTERN.match(folder).groups()[0]
    return 'joints19' if pose_model == '_coco19' else 'joints15'


def is_sample_file(file):
    return SAMPLE_ID.match(file) is not None


def get_sample_id(file):
    id_str = SAMPLE_ID.match(file).groups()[0]
    return int(id_str)


def is_video_file(file):
    return VIDEO_FILE_PATTERN.match(file) is not None


def get_camera_id(file):
    match = VIDEO_FILE_PATTERN.match(file)
    return -1 if match is None else int(match.groups()[0])


def make_df_columns(pose_model, has_z=True):
    joint_data_keys = ['j{}x', 'j{}y', 'j{}z', 'j{}c'] if has_z else ['j{}x', 'j{}y', 'j{}c']
    columns = ['sample_id', 'person_id']
    n_joints = int(pose_model.strip('joints'))
    for n in range(n_joints):
        columns += list(map(lambda x: x.format(n), joint_data_keys))
    return columns


def load_calibrations_pb(calibrations_file, referencial=9999, cameras=None):
    """
    Load Panoptic CMU dataset calibrations from HD cameras, convert in to a
    is_msgs.camera_pb2.CameraCalibration protobuf.
    """

    with open(calibrations_file, 'r') as f:
        calibrations = json.load(f)['cameras']

    calibrations = list(filter(lambda d: d['type'] == 'hd', calibrations))
    calibrations_pb = {}
    for calibration in calibrations:
        calib_pb = CameraCalibration()

        camera_id = int(calibration['name'].split('_')[1])
        if cameras is not None and camera_id not in cameras:
            continue

        width, height = calibration['resolution'][0], calibration['resolution'][1]
        intrinsic = np.array(calibration['K'])
        distortion = np.array(calibration['distCoef'])[:, np.newaxis]
        R = np.array(calibration['R'])
        t = np.array(calibration['t'])
        extrinsic = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))

        calib_pb.id = camera_id
        calib_pb.resolution.width = width
        calib_pb.resolution.height = height
        calib_pb.intrinsic.CopyFrom(to_tensor(intrinsic))
        calib_pb.distortion.CopyFrom(to_tensor(distortion))
        ext = calib_pb.extrinsic.add()
        setattr(ext, 'from', referencial)
        setattr(ext, 'to', camera_id)
        ext.tf.CopyFrom(to_tensor(extrinsic))

        calibrations_pb[camera_id] = calib_pb

    return calibrations_pb
