import json
from argparse import ArgumentParser
from os import walk
from os.path import join, dirname, basename, exists
import numpy as np
from itertools import combinations
from collections import defaultdict

from src.panoptic_dataset.utils import is_sequence_folder
from src.utils.logger import Logger

log = Logger(name='GrouperDuration')

def main(experiment_folders, output_folder):

    durations = defaultdict(list)

    for experiment_folder in experiment_folders:
        exp_name = basename(dirname(experiment_folder + '/'))
        _, exp_seq_folders, _ = next(walk(experiment_folder))
        exp_seq_folders = list(sorted(filter(is_sequence_folder, exp_seq_folders)))

        for exp_seq_folder in exp_seq_folders:
            exp_seq_folder_path = join(experiment_folder, exp_seq_folder)
            _, pose_model_folders, _ = next(walk(exp_seq_folder_path))

            for pose_model_folder in pose_model_folders:

                log.info("{} | {} | {}", exp_name, exp_seq_folder, pose_model_folder)

                trace_file_path = join(exp_seq_folder_path, pose_model_folder, 'trace.json')
                with open(trace_file_path, 'r') as f:
                    trace_data = json.load(f)

                for span in trace_data:
                    if 'detections' not in span:
                        continue

                    detections = span['detections']
                    cameras = detections.keys()
                    if len(cameras) < 2:
                        continue

                    n_computations = sum([
                        detections[cam0] * detections[cam1]
                        for cam0, cam1 in combinations(cameras, 2)
                    ])
                    durations[n_computations].append(span['duration'])

    output_file_path = join(output_folder, 'grouper_durations.json')
    log.info("Saving results on '{}'", output_file_path)
    with open(output_file_path, 'w') as f:
        json.dump(durations, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    parser = ArgumentParser()
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

    args = parser.parse_args()
    main(experiment_folders=args.experiment_folders, output_folder=args.output_folder)
