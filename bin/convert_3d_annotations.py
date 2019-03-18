import json
from argparse import ArgumentParser
from os import makedirs, walk
from os.path import join, dirname, exists
from shutil import rmtree
import numpy as np
import pandas as pd

from src.panoptic_dataset.utils import is_sequence_folder, is_pose_folder, get_joints_key
from src.panoptic_dataset.utils import is_sample_file, get_sample_id, make_df_columns
from src.utils.logger import Logger

BASEDIR = join(dirname(__file__))

log = Logger(name='Convert3DAnnotations')


def main(dataset_folder, output_folder):

    if not exists(dataset_folder):
        raise Exception("'{}' folder doesn't exist.".format(dataset_folder))

    if output_folder is None:
        output_folder = dataset_folder

    _, sequence_folders, _ = next(walk(dataset_folder))
    sequence_folders = list(filter(is_sequence_folder, sequence_folders))

    for s_folder in sequence_folders:
        sequence_path = join(dataset_folder, s_folder)
        _, pose_folders, _ = next(walk(sequence_path))
        pose_folders = list(filter(is_pose_folder, pose_folders))

        for p_folder in pose_folders:
            pose_folder_path = join(dataset_folder, s_folder, p_folder)
            _, _, pose_files = next(walk(pose_folder_path))
            pose_files = list(sorted(filter(is_sample_file, pose_files)))

            joints_key = get_joints_key(p_folder)
            all_joints_data = []
            for p_file in pose_files:
                pose_file_path = join(pose_folder_path, p_file)
                with open(pose_file_path, 'r') as f:
                    try:
                        pose_data = json.load(f)
                    except json.decoder.JSONDecodeError as e:
                        log.warn("Failed to load file: \n{}\nReason: {}", pose_file_path, str(e))

                n_bodies = len(pose_data['bodies'])
                if n_bodies == 0:
                    continue

                joints_data = []
                person_ids = []
                for p_data in pose_data['bodies']:
                    joints_data.append(np.array(p_data[joints_key]))
                    person_ids.append(p_data['id'])

                sample_id = get_sample_id(p_file)
                sample_id_array = sample_id * np.ones(n_bodies).reshape(n_bodies, 1)
                person_ids = np.array(person_ids).reshape(n_bodies, 1)

                joints_data = np.hstack((sample_id_array, person_ids, np.vstack(joints_data)))
                all_joints_data.append(joints_data)

            all_joints_data = np.vstack(all_joints_data)
            df = pd.DataFrame(data=all_joints_data, columns=make_df_columns(joints_key))

            output_folder_path = join(output_folder, s_folder, '3d_annotations', joints_key)
            if exists(output_folder_path):
                rmtree(output_folder_path)
            makedirs(output_folder_path)

            log.info("Writing data from sequence {} pose model {}", s_folder, joints_key)

            annotations_file_path = join(output_folder_path, 'data.csv')
            df.to_csv(path_or_buf=annotations_file_path, header=True, index=False)

            info_file_path = join(output_folder_path, 'info.json')
            info = {
                'begin': get_sample_id(pose_files[0]),
                'end': get_sample_id(pose_files[-1]),
                'n_person': np.unique(df['person_id']).size,
            }
            with open(info_file_path, 'w') as f:
                json.dump(info, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '--dataset-folder',
        type=str,
        required=True,
        help="""Path to folder containing CMU Panoptic folders.
        Each sequence folder must contain either 'hdPose3d_stage1' 
        or 'hdPose3d_stage1_coco19' folder. Each one represent a 
        different pose model. All present folders will be processed.""")
    parser.add_argument(
        '--output-folder',
        type=str,
        required=False,
        help="""Path to folder to write the CSV files for each sequence
        and pose model. A folder named '3d_annotations' will be created,
        and inside that will be created folders with same name as original.
        If not specified, these files will be saved on the sequence folder 
        on a 'calibrations' folder.""")

    args = parser.parse_args()
    main(args.dataset_folder, args.output_folder)