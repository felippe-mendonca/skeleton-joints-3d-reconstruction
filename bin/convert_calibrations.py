import re
import json
from argparse import ArgumentParser
from os import makedirs, walk
from os.path import join, dirname, exists
from shutil import rmtree
from google.protobuf.json_format import MessageToDict

from panoptic_dataset.utils import load_calibrations_pb
from utils.logger import Logger

BASEDIR = join(dirname(__file__))
SEQUENCE_PATTERN = re.compile(r'^[0-9]{6}_[a-zA-Z]+[0-9]{1}$')

log = Logger(name='ConvertCalibs')


def is_sequence_folder(s):
    return SEQUENCE_PATTERN.match(s) is not None


def main(dataset_folder, output_folder):

    if not exists(dataset_folder):
        raise Exception("'{}' folder doesn't exist.".format(dataset_folder))

    _, sequence_folders, _ = next(walk(dataset_folder))
    sequence_folders = list(filter(is_sequence_folder, sequence_folders))

    for s_folder in sequence_folders:
        calib_file = join(dataset_folder, s_folder, 'calibration.json')
        if not exists(calib_file):
            log.warn("Calibration file from {} sequence doesn't exist. Skipping.", s_folder)
            continue

        if output_folder is None:
            output_folder = dataset_folder
        s_calibs_folder = join(output_folder, s_folder, 'calibrations')

        if exists(s_calibs_folder):
            rmtree(s_calibs_folder)
        makedirs(s_calibs_folder)

        calibs = load_calibrations_pb(calib_file, referencial=9999)
        for camera_id, calib in calibs.items():
            calib_dict = MessageToDict(
                message=calib,
                including_default_value_fields=True,
                preserving_proto_field_name=True)
            calib_pb_file = join(s_calibs_folder, '{}.json'.format(camera_id))

            log.info("Saving '{}'", calib_pb_file)
            with open(calib_pb_file, 'w') as f:
                json.dump(calib_dict, f, indent=True, sort_keys=True)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '--dataset-folder',
        type=str,
        required=True,
        help="""Path to folder containing CMU Panoptic folders.
        Each sequence folder must contain a 'calibration.json' file.""")
    parser.add_argument(
        '--output-folder',
        type=str,
        required=False,
        help="""Path to folder to write the JSON files containing a 
        calibration of each camera, following the is_msgs.camera_pb2.CameraCalibration 
        protobuf schema. If not specified, these files will be saved on the 
        sequence folder on a 'calibrations' folder.""")

    args = parser.parse_args()
    main(args.dataset_folder, args.output_folder)