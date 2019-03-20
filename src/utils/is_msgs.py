import json
from google.protobuf.json_format import ParseDict
from is_msgs.camera_pb2 import CameraCalibration
from is_msgs.image_pb2 import ObjectAnnotations
from is_msgs.image_pb2 import HumanKeypoints as HKP

from src.panoptic_dataset.joints import index_to_human_keypoint


def load_camera_calibration(file):
    with open(file, 'r') as f:
        calib_dict = json.load(f)
        return ParseDict(calib_dict, CameraCalibration())


def data_frame_to_object_annotations(annotations, model, has_z=False, frame_id=0, resolution=None):

    if model != 'joints15' and model != 'joints19':
        raise Exception("Invalid Model passed. Can be either 'joints15' or 'joints19'")

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
            kp_id = index_to_human_keypoint(joint_id, model)

            # check for invalid joint
            if (x == 0.0 and y == 0.0 and z == 0.0) or c < 0.0:
                continue
            
            if kp_id == HKP.Value('UNKNOWN_HUMAN_KEYPOINT'):
                continue

            keypoint = skeleton.keypoints.add()
            keypoint.position.x = x
            keypoint.position.y = y
            keypoint.position.z = z
            keypoint.score = c
            keypoint.id = index_to_human_keypoint(joint_id, model)

    return annotations_pb
