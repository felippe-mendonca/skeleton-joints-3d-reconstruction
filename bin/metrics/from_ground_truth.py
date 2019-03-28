import json
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from os import makedirs, walk
from os.path import join, dirname, basename, exists
from shutil import rmtree

from src.panoptic_dataset.utils import is_sequence_folder
from src.panoptic_dataset.joints import index_to_human_keypoint
from src.utils.logger import Logger
from src.utils.metrics import error_per_joint

log = Logger(name="MetricsFromGroundTruth")


def main(dataset_folder, experiment_folders):

    np.set_printoptions(precision=2)
    ratios = []

    gt_number_individuals_global, exp_number_individuals_global = 0, 0
    errors_global, errors_n_samples_global = [], []
    setup = []

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
                exp_data = exp_data[exp_data['person_id'] >= 0]

                gt_data_file_path = join(gt_data_folder_path, pose_model_folder, 'data.csv')
                gt_data = pd.read_csv(gt_data_file_path)

                gt_number_individuals = len(gt_data.index)
                exp_number_individuals = len(exp_data.index) 

                gt_number_individuals_global += gt_number_individuals
                exp_number_individuals_global += exp_number_individuals

                ratio_number_individuals = exp_number_individuals / gt_number_individuals 
                ratios.append("${:.2f}$".format(100 * ratio_number_individuals))
                log.info(
                    "{} | {} | {} / {} = {:.2%}",
                    exp_name,
                    exp_seq_folder,
                    exp_number_individuals,
                    gt_number_individuals,
                    ratio_number_individuals
                )

                for sample_id in range_sample_id:
                    _gt_data = gt_data[gt_data['sample_id'] == sample_id]
                    _exp_data = exp_data[exp_data['sample_id'] == sample_id]
                    for _, exp_ind in _exp_data.iterrows():
                        gt_ind = _gt_data[_gt_data['person_id'] == exp_ind['person_id']].iloc[0]
                        error = error_per_joint(gt_ind, exp_ind, pose_model=pose_model_folder)
                        errors.append(error)

                errors = np.vstack(errors)
                errors_n_samples = np.sum(~np.isnan(errors), axis=0)
                errors = np.nanmean(errors, axis=0)

                errors_global.append(errors)
                errors_n_samples_global.append(errors_n_samples)
                setup.append((exp_name, exp_seq_folder))

    
    errors_global = np.vstack(errors_global)
    errors_n_samples_global = np.vstack(errors_n_samples_global)

    print(errors_global)
    print(errors_n_samples_global)
    print(setup)

    print(" & ".join(ratios).replace('.', ','))
    print("{:.2%}".format(exp_number_individuals_global / gt_number_individuals_global))


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

    args = parser.parse_args()
    main(dataset_folder=args.dataset_folder, experiment_folders=args.experiment_folders)