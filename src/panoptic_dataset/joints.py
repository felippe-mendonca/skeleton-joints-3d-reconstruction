from is_msgs.image_pb2 import HumanKeypoints as HKP


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

    INVALID_INDEX_MSG = "Invalid index for model {}. Must be less then {}"

    if model == 'joints15':
        keypoints = MODEL_15_JOINTS
    elif model == 'joints19':
        keypoints = MODEL_19_JOINTS
    else:
        raise Exception("Invalid Model passed. Can be either 'joints15' or 'joints19'")

    n_joints = len(keypoints)
    if index >= n_joints:
        raise Exception(INVALID_INDEX_MSG.format(model, n_joints))

    return keypoints[index]


def get_joint_links(model):
    if model == 'joints15':
        return MODEL_15_LINKS
    elif model == 'joints19':
        return MODEL_19_LINKS
    else:
        raise Exception("Invalid Model passed. Can be either 'joints15' or 'joints19'")
