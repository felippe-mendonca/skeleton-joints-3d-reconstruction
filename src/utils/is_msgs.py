import json
from google.protobuf.json_format import ParseDict
from is_msgs.camera_pb2 import CameraCalibration


def load_camera_calibration(file):
    with open(file, 'r') as f:
        calib_dict = json.load(f)
        return ParseDict(calib_dict, CameraCalibration())