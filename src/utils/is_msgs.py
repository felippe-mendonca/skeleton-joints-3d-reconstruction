import json
import numpy as np
from google.protobuf.json_format import ParseDict
from is_msgs.camera_pb2 import CameraCalibration
from is_msgs.image_pb2 import ObjectAnnotations
from is_msgs.image_pb2 import HumanKeypoints as HKP

from src.panoptic_dataset.utils import is_valid_model
from src.panoptic_dataset.joints import index_to_human_keypoint, human_keypoint_to_index


def load_camera_calibration(file):
    with open(file, 'r') as f:
        calib_dict = json.load(f)
        return ParseDict(calib_dict, CameraCalibration())


def data_frame_to_object_annotations(annotations, model, has_z=False, frame_id=0, resolution=None):

    is_valid_model(model)

    if annotations.empty:
        return ObjectAnnotations()

    n_model_joints = int(model.strip('joints'))
    joints_values = len(annotations.drop(['sample_id', 'person_id'], axis=1).columns)
    n_annotations_joints = int(joints_values / (4 if has_z else 3))
    n_joints = min(n_model_joints, n_annotations_joints)

    annotations_pb = ObjectAnnotations()
    annotations_pb.frame_id = frame_id
    if resolution is not None:
        annotations_pb.resolution.width = resolution[0]
        annotations_pb.resolution.height = resolution[1]

    for _, annotation in annotations.iterrows():
        skeleton = annotations_pb.objects.add()
        skeleton.id = int(annotation['person_id'])

        for joint_id in range(n_joints):
            x = annotation['j{:d}x'.format(joint_id)]
            y = annotation['j{:d}y'.format(joint_id)]
            z = annotation['j{:d}z'.format(joint_id)] if has_z else 0.0
            c = annotation['j{:d}c'.format(joint_id)]
            human_keypoint = index_to_human_keypoint(joint_id, model)

            # check for invalid joint
            if (x == 0.0 and y == 0.0 and z == 0.0) or c < 0.0:
                continue

            if human_keypoint == HKP.Value('UNKNOWN_HUMAN_KEYPOINT'):
                continue

            keypoint = skeleton.keypoints.add()
            keypoint.position.x = x
            keypoint.position.y = y
            keypoint.position.z = z
            keypoint.score = c
            keypoint.id = human_keypoint

    return annotations_pb


def object_annotations_to_np(annotations_pb,
                             model,
                             has_z=False,
                             add_person_id=False,
                             sample_id=None):

    is_valid_model(model)

    n_model_joints = int(model.strip('joints'))
    data_offset = (1 if add_person_id else 0) + (1 if sample_id is not None else 0)
    n_joint_data = (4 if has_z else 3)
    n_cols = n_model_joints * n_joint_data + data_offset

    n_skeletons = len(annotations_pb.objects)
    annotations = np.zeros((n_skeletons, n_cols), dtype=np.float)
    annotations[:, (data_offset + n_joint_data - 1)::(n_joint_data)] = -1

    for row, skeleton in enumerate(annotations_pb.objects):
        for keypoint in skeleton.keypoints:
            joint_id = human_keypoint_to_index(human_keypoint=keypoint.id, model=model)

            col = joint_id * n_joint_data + data_offset
            annotations[row, col + 0] = keypoint.position.x
            annotations[row, col + 1] = keypoint.position.y
            if has_z:
                annotations[row, col + 2] = keypoint.position.z
            annotations[row, col + (n_joint_data - 1)] = keypoint.score

        if sample_id is not None:
            annotations[row, 0] = sample_id
            if add_person_id:
                annotations[row, 1] = skeleton.id
        elif add_person_id:
            annotations[row, 0] = skeleton.id

    return annotations
