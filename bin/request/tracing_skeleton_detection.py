import re
import json
from argparse import ArgumentParser
from requests import get
from os import walk
from os.path import join, exists
import pandas as pd

from src.panoptic_dataset.utils import is_sequence_folder, AVAILABLE_MODELS
from src.utils.logger import Logger

log = Logger(name='GetTracing')


def is_exp_folder(folder):
    return re.match('^exp[0-9]+$', folder) is not None


gpu_device_ids = ['10de1b80', '10de1b81', '10de1b06', '10de1b02']

ZIPKIN_ENDPOINT = "{uri}/zipkin/api/v2/traces?" \
                + "annotationQuery=gpu_device_id%3D{gpu_id}&limit={limit}" \
                + "&serviceName=skeletonsdetector&spanName=skeletonsdetector.detect"


def main(experiment_folder, n_samples, zipkin_uri):
    def get_duration(trace):
        duration = -1
        for span in trace:
            if span['name'] == 'skeletonsdetector.detect':
                duration = int(span['duration'])
        return duration

    durations_data = {}
    for gpu_device_id in gpu_device_ids:
        endpoint = ZIPKIN_ENDPOINT.format(uri=zipkin_uri, gpu_id=gpu_device_id, limit=n_samples)
        print(endpoint)
        reply = get(endpoint)
        tracing_data = json.loads(reply.text)
        durations_data[gpu_device_id] = list(map(get_duration, tracing_data))

    df = pd.DataFrame(data=durations_data)
    output_file_path = join(experiment_folder, 'detection_durations.csv')
    df.to_csv(output_file_path, header=True, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--experiment-folder',
        type=str,
        required=True,
        help="""Folder to save output tracing file.""")
    parser.add_argument(
        '--n-samples',
        type=int,
        required=False,
        default=10000,
        help="Number of sample for each gpu device to be collected.")
    parser.add_argument('--zipkin-uri', type=str, required=True, help="""Zipkin URI""")

    args = parser.parse_args()
    main(
        experiment_folder=args.experiment_folder,
        n_samples=args.n_samples,
        zipkin_uri=args.zipkin_uri)
