import re
import json
from argparse import ArgumentParser
from requests import get
from os import walk
from os.path import join, exists

from src.panoptic_dataset.utils import is_sequence_folder, AVAILABLE_MODELS
from src.utils.logger import Logger

log = Logger(name='GetTracing')


def is_exp_folder(folder):
    return re.match('^exp[0-9]+$', folder) is not None


ZIPKIN_ENDPOINT = "{uri}/zipkin/api/v2/traces?" \
                + "annotationQuery=experiment%3D{exp}%20and%20sequence%3D{sequence}" \
                + "&limit=15000&serviceName=requestskeletonslocalization&spanName=request"


def main(experiments_folder, zipkin_uri):

    _, exp_folders, _ = next(walk(experiments_folder))
    exp_folders = list(filter(is_exp_folder, exp_folders))

    for exp_folder in exp_folders:
        exp_folder_path = join(experiments_folder, exp_folder)
        _, sequence_folders, _ = next(walk(exp_folder_path))
        sequence_folders = list(filter(is_sequence_folder, sequence_folders))

        for sequence_folder in sequence_folders:
            sequence_folder_path = join(exp_folder_path, sequence_folder)
            _, pose_folders, _ = next(walk(sequence_folder_path))
            pose_folders = list(filter(lambda x: x in AVAILABLE_MODELS, pose_folders))

            for pose_folder in pose_folders:
                data_file_path = join(sequence_folder_path, pose_folder, 'data.csv')
                if not exists(data_file_path):
                    continue

                endpoint = ZIPKIN_ENDPOINT.format(
                    uri=zipkin_uri, exp=exp_folder, sequence=sequence_folder)
                reply = get(endpoint)
                tracing_data = json.loads(reply.text)

                def save_trace(tracing, filename):
                    trace_output_file_path = join(sequence_folder_path, pose_folder, filename)
                    log.info("Saving tracing on '{}'", trace_output_file_path)
                    with open(trace_output_file_path, 'w') as f:
                        json.dump(tracing, f)

                for trace in tracing_data:
                    for span in trace:
                        if span['localEndpoint']['serviceName'] == 'skeletonsgrouper':
                            detections_str = re.sub(r'([0-9]+):', r'"\1":',
                                                    span['tags']['detections'])
                            span['tags']['detections'] = json.loads(detections_str)

                save_trace(tracing_data, 'raw_trace.json')

                def reduce_trace(trace):
                    reduced_trace = {}
                    for span in trace:
                        if span['localEndpoint']['serviceName'] == 'skeletonsgrouper':
                            reduced_trace['detections'] = span['tags']['detections']
                            reduced_trace['duration'] = span['duration']
                        else:
                            reduced_trace['sample_id'] = span['tags']['sample_id']
                    return reduced_trace

                reduced_trace = list(map(reduce_trace, tracing_data))

                save_trace(reduced_trace, 'trace.json')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--experiments-folder',
        type=str,
        required=True,
        help="""Folder containing the data saved by request.skeletons_localization 
        script. For each sequence of each experiments, a JSON file containing the 
        tracing will be saved on that folder.""")
    parser.add_argument('--zipkin-uri', type=str, required=True, help="""Zipkin URI""")

    args = parser.parse_args()
    main(experiments_folder=args.experiments_folder, zipkin_uri=args.zipkin_uri)
