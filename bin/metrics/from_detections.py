import json
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from os import makedirs, walk
from os.path import join, dirname, basename, exists
from itertools import product

from is_msgs.image_pb2 import HumanKeypoints as HKP
from src.panoptic_dataset.utils import is_sequence_folder
from src.panoptic_dataset.joints import index_to_human_keypoint
from src.utils.logger import Logger
from src.utils.metrics import error_per_joint, possible_groups

log = Logger(name="MetricsFromDetections")


def main(dataset_folder, experiment_folders, output_folder, output_prefix):

    np.set_printoptions(precision=2)
    pd.set_option('precision', 2)

    errors_global, errors_n_samples_global = [], []
    output_data = {'experiment': [], 'sequence': [], 'g_ind': []}

    for experiment_folder in experiment_folders:
        exp_name = basename(dirname(experiment_folder + '/'))
        _, exp_seq_folders, _ = next(walk(experiment_folder))
        exp_seq_folders = list(sorted(filter(is_sequence_folder, exp_seq_folders)))

        for exp_seq_folder in exp_seq_folders:
            exp_seq_folder_path = join(experiment_folder, exp_seq_folder)
            _, pose_model_folders, _ = next(walk(exp_seq_folder_path))

            info_file_path = join(dataset_folder, exp_seq_folder, 'info.json')
            with open(info_file_path, 'r') as f:
                info_data = json.load(f)
            range_sample_id = range(info_data['begin'], info_data['end'] + 1)

            gt_data_folder_path = join(dataset_folder, exp_seq_folder, '3d_annotations')
            errors = []
            for pose_model_folder in pose_model_folders:
                exp_data_folder_path = join(exp_seq_folder_path, pose_model_folder)
                exp_data_file_path = join(exp_data_folder_path, 'data.csv')
                exp_data = pd.read_csv(exp_data_file_path)

                gt_data_file_path = join(gt_data_folder_path, pose_model_folder, 'data.csv')
                gt_data = pd.read_csv(gt_data_file_path)

                for sample_id in range_sample_id:
                    _gt_data = gt_data[gt_data['sample_id'] == sample_id]
                    _exp_data = exp_data[exp_data['sample_id'] == sample_id]

                    _gt_its, _exp_its = _gt_data.index, _exp_data.index
                    error_pairs = {
                        pair: error_per_joint(
                            gt_data.iloc[pair[0]],
                            exp_data.iloc[pair[1]],
                            pose_model=pose_model_folder)
                        for pair in product(_gt_its, _exp_its)
                    }

                    def group_error(group):
                        return sum(map(lambda x: np.nanmean(error_pairs[x]), group))

                    best_group = min(possible_groups(_gt_its, _exp_its), key=group_error)
                    if len(best_group) == 0:
                        continue

                    sample_errors = np.vstack(map(lambda x: error_pairs[x], best_group))
                    errors.append(sample_errors)

                errors = np.vstack(errors)

                gt_number_individuals = len(gt_data.index)
                exp_number_individuals = errors.shape[0]

                ratio_number_individuals = exp_number_individuals / gt_number_individuals
                output_data['experiment'].append(exp_name)
                output_data['sequence'].append(exp_seq_folder)
                output_data['g_ind'].append(100.0 * ratio_number_individuals)

                errors_n_samples = np.sum(~np.isnan(errors), axis=0)
                errors = np.nanmean(errors, axis=0)

                errors_global.append(errors)
                errors_n_samples_global.append(errors_n_samples)

    errors_global = np.vstack(errors_global)
    errors_n_samples_global = np.vstack(errors_n_samples_global)

    errors = 10.0 * np.sum( errors_global * errors_n_samples_global, axis=0) \
      / np.sum( errors_n_samples_global, axis=0)

    def human_kp(index):
        return index_to_human_keypoint(index, 'joints19')

    errors_columns = list(map(HKP.Name, map(human_kp, range(errors.size))))
    df = pd.DataFrame(data=errors[np.newaxis, :], columns=errors_columns)
    errors_file_path = join(output_folder, '{}_joint_errors.csv'.format(output_prefix))
    df.to_csv(errors_file_path, header=True, index=False)
    print(df.T)

    df = pd.DataFrame(data=output_data)
    output_data_file_path = join(output_folder, '{}_grouped_person.csv'.format(output_prefix))
    df.to_csv(output_data_file_path, header=True, index=False)
    print(df)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset-folder',
        type=str,
        required=True,
        help="""Path to folder containing CMU Panoptic folders. To be able to compute 
        all metrics, 3D annotations from the same pose model must be present.""")
    parser.add_argument(
        '--experiment-folders',
        type=str,
        required=True,
        nargs='+',
        help="""Path to folders where experiment output data was saved. All 
        sequence folder will be evaluated, as well as different model pose.""")
    parser.add_argument(
        '--output-folder',
        type=str,
        required=True,
        help="""Path to folder to save CSV files with results.""")
    parser.add_argument(
        '--output-prefix',
        type=str,
        default="",
        help="""Prefix string to be added on output files.""")

    args = parser.parse_args()
    main(
        dataset_folder=args.dataset_folder,
        experiment_folders=args.experiment_folders,
        output_prefix=args.output_prefix,
        output_folder=args.output_folder)