import json
from os.path import join, exists, dirname
import numpy as np

from is_msgs.camera_pb2 import CameraCalibration
from utils.numpy import to_tensor


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
