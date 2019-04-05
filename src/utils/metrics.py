import numpy as np
import pandas as pd
from itertools import combinations, permutations, product
from src.panoptic_dataset.utils import is_valid_model


def shape_data(data):
    if type(data) in [pd.DataFrame, pd.Series]:
        data = data.values
    data = data[2:].reshape(-1, 4).T
    return data


def invalid_joints(data):
    null_joints = (data[0:3, :] == 0.0).all(axis=0)
    untrusted_joints = data[3, :] < 0
    return np.logical_or(null_joints, untrusted_joints)


def compute_error_per_joint(gt, exp):
    error = np.sqrt(np.sum(np.power(gt[0:3, :] - exp[0:3, :], 2), axis=0))
    invalid_error = np.logical_or(invalid_joints(gt), invalid_joints(exp))
    error[invalid_error] = np.nan
    return error


def error_per_joint(gt_data, exp_data, pose_model):

    is_valid_model(pose_model)

    gt_data = shape_data(gt_data)
    exp_data = shape_data(exp_data)

    return compute_error_per_joint(gt_data, exp_data)


def zip_groups(combs):
    return map(list, (map(lambda x: zip(*x), combs)))


def possible_groups(gt_its, exp_its):
    n_pairs = min(len(exp_its), len(gt_its))
    gt_it_combs = combinations(gt_its, n_pairs)
    exp_it_perms = permutations(exp_its, n_pairs)
    return list(zip_groups(product(gt_it_combs, exp_it_perms)))