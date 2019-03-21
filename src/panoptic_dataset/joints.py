from src.panoptic_dataset.utils import is_valid_model
from is_msgs.image_pb2 import HumanKeypoints as HKP


def _reverse_model_joints(joints_list):
    reverse_list = [-1] * (max(joints_list) + 1)
    for joint_index, human_keypoint in enumerate(joints_list):
        reverse_list[human_keypoint] = joint_index

    return reverse_list

MODEL_15_JOINTS = [
    HKP.Value('NECK'),           \
    HKP.Value('HEAD'),           \
    HKP.Value('CHEST'),          \
    HKP.Value('LEFT_SHOULDER'),  \
    HKP.Value('LEFT_ELBOW'),     \
    HKP.Value('LEFT_WRIST'),     \
    HKP.Value('LEFT_HIP'),       \
    HKP.Value('LEFT_KNEE'),      \
    HKP.Value('LEFT_ANKLE'),     \
    HKP.Value('RIGHT_SHOULDER'), \
    HKP.Value('RIGHT_ELBOW'),    \
    HKP.Value('RIGHT_WRIST'),    \
    HKP.Value('RIGHT_HIP'),      \
    HKP.Value('RIGHT_KNEE'),     \
    HKP.Value('RIGHT_ANKLE')
]

# HumanKeyPoint.Value('UNKNOWN_HUMAN_KEYPOINT') represents the Background
MODEL_19_JOINTS = [
    HKP.Value('NECK'),                   \
    HKP.Value('NOSE'),                   \
    HKP.Value('UNKNOWN_HUMAN_KEYPOINT'), \
    HKP.Value('LEFT_SHOULDER'),          \
    HKP.Value('LEFT_ELBOW'),             \
    HKP.Value('LEFT_WRIST'),             \
    HKP.Value('LEFT_HIP'),               \
    HKP.Value('LEFT_KNEE'),              \
    HKP.Value('LEFT_ANKLE'),             \
    HKP.Value('RIGHT_SHOULDER'),         \
    HKP.Value('RIGHT_ELBOW'),            \
    HKP.Value('RIGHT_WRIST'),            \
    HKP.Value('RIGHT_HIP'),              \
    HKP.Value('RIGHT_KNEE'),             \
    HKP.Value('RIGHT_ANKLE'),            \
    HKP.Value('LEFT_EYE'),               \
    HKP.Value('LEFT_EAR'),               \
    HKP.Value('RIGHT_EYE'),              \
    HKP.Value('RIGHT_EAR')
]

MODEL_15_JOINTS_REVERSED = _reverse_model_joints(MODEL_15_JOINTS)
MODEL_19_JOINTS_REVERSED = _reverse_model_joints(MODEL_19_JOINTS)

MODEL_15_LINKS = []

MODEL_19_LINKS = [
    (HKP.Value('NECK'), HKP.Value('LEFT_SHOULDER')),
    (HKP.Value('LEFT_SHOULDER'), HKP.Value('LEFT_ELBOW')),
    (HKP.Value('LEFT_ELBOW'), HKP.Value('LEFT_WRIST')),
    (HKP.Value('NECK'), HKP.Value('LEFT_HIP')),
    (HKP.Value('LEFT_HIP'), HKP.Value('LEFT_KNEE')),
    (HKP.Value('LEFT_KNEE'), HKP.Value('LEFT_ANKLE')),
    (HKP.Value('NECK'), HKP.Value('RIGHT_SHOULDER')),
    (HKP.Value('RIGHT_SHOULDER'), HKP.Value('RIGHT_ELBOW')),
    (HKP.Value('RIGHT_ELBOW'), HKP.Value('RIGHT_WRIST')),
    (HKP.Value('NECK'), HKP.Value('RIGHT_HIP')),
    (HKP.Value('RIGHT_HIP'), HKP.Value('RIGHT_KNEE')),
    (HKP.Value('RIGHT_KNEE'), HKP.Value('RIGHT_ANKLE')),
    (HKP.Value('NOSE'), HKP.Value('LEFT_EYE')),
    (HKP.Value('LEFT_EYE'), HKP.Value('LEFT_EAR')),
    (HKP.Value('NOSE'), HKP.Value('RIGHT_EYE')),
    (HKP.Value('RIGHT_EYE'), HKP.Value('RIGHT_EAR')),
]


def index_to_human_keypoint(index, model):

    is_valid_model(model)

    INVALID_INDEX_MSG = "Invalid index for model {}. Must be less then {}"

    if model == 'joints15':
        human_keypoints = MODEL_15_JOINTS
    elif model == 'joints19':
        human_keypoints = MODEL_19_JOINTS

    n_joints = len(human_keypoints)
    if index >= n_joints:
        raise Exception(INVALID_INDEX_MSG.format(model, n_joints))

    return human_keypoints[index]


def human_keypoint_to_index(human_keypoint, model):

    is_valid_model(model)

    INVALID_HUMAN_KEYPOINT_MSG = "Invalid HumanKeypoint for model {}."

    if model == 'joints15':
        joint_indexes = MODEL_15_JOINTS_REVERSED
    elif model == 'joints19':
        joint_indexes = MODEL_19_JOINTS_REVERSED

    joint_index = joint_indexes[human_keypoint]
    if joint_index == -1:
        raise Exception(INVALID_HUMAN_KEYPOINT_MSG.format(model))

    return joint_index


def get_joint_links(model):
    if model == 'joints15':
        return MODEL_15_LINKS
    elif model == 'joints19':
        return MODEL_19_LINKS
    else:
        raise Exception("Invalid Model passed. Can be either 'joints15' or 'joints19'")
