import numpy as np
import pandas as pd
from src.panoptic_dataset.utils import is_valid_model


def error_per_joint(gt_data, exp_data, pose_model):

    is_valid_model(pose_model)

    def shape_data(data):
        if type(data) in [pd.DataFrame, pd.Series]:
            data = data.values
        data = data[2:].reshape(-1, 4).T
        return data

    gt_data = shape_data(gt_data)
    exp_data = shape_data(exp_data)

    def invalid_joints(data):
        null_joints = (data[0:3, :] == 0.0).all(axis=0)
        untrusted_joints = data[3, :] < 0
        return np.logical_or(null_joints, untrusted_joints)

    error = np.sqrt(np.sum(np.power(gt_data[0:3, :] - exp_data[0:3, :], 2), axis=0))
    invalid_error = np.logical_or(invalid_joints(gt_data), invalid_joints(exp_data))
    error[invalid_error] = np.nan

    return error
