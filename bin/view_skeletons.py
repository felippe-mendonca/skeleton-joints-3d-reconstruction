from argparse import ArgumentParser
import cv2
import pandas as pd

from src.utils.is_msgs import data_frame_to_object_annotations
from src.panoptic_dataset.joints import get_joint_links
from src.utils.drawing import draw_skeletons


def main(video_file, annotations_file, resize_factor, model):
    vc = cv2.VideoCapture(video_file)
    is_paused = False

    frame_id = 0
    joint_links = get_joint_links(model)

    annotations_data = pd.read_csv(annotations_file)

    while vc.isOpened():
        if not is_paused:
            has_frame, frame = vc.read()
            if not has_frame:
                break

            frame_annotations = annotations_data[annotations_data['sample_id'] == frame_id]
            obj_annotations = data_frame_to_object_annotations(frame_annotations, model)
            frame = draw_skeletons(frame, obj_annotations, joint_links)

            if resize_factor is not None and resize_factor < 1.0 and resize_factor > 0.0:
                frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)

            cv2.imshow("Press [q] to quit and [k] to pause/resume", frame)

            frame_id += 1

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('k'):
            is_paused = ~is_paused


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help="""Path to video file. Make sure that you are specifing the correct 
        annotation file to this video, otherwise you will see wierd stuff on 
        your screen. (By the way, try it once! It's funny!)""")
    parser.add_argument(
        '--annotations',
        type=str,
        required=False,
        help="""Path to folder to CSV with annotations file.""")
    parser.add_argument(
        '--model',
        type=str,
        default='joints19',
        help=""""Pose model to be used during skeleton drawing and conversion 
        to is_msgs.image_pb2.ObjectAnnotations conversion. Can be either 
        'joints15' or 'joints19'. If it's not specified, 'joints19' will be used.
        In case of a wrong given model, if it has less joints, some of them will 
        be not displayed.""")
    parser.add_argument(
        '--resize',
        type=float,
        required=False,
        help=""""Scale factor to be applied on image before display it.""")

    args = parser.parse_args()
    main(args.video, args.annotations, args.resize, args.model)