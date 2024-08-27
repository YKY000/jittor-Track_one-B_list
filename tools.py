import os
from matplotlib import pyplot as plt
import random
import numpy as np
import jittor
import pickle

def save_features(features, four_dimensional_features, features_path):
    """
    临时将提取的训练图像特征保存到文件中。

    参数:
    features (dict): 存储图像特征的字典。
    features_path (str): 保存特征的文件路径。 Dataset/features.pkl
    """
    with open(features_path, 'wb') as f:
        pickle.dump(features, f)

    print(f"特征已保存到 {features_path} ")


def load_features(features_path):
    """
    从文件中加载特征

    参数:
    features_path (str): 包含特征的文件路径。

    返回:
    dict: 加载的特征字典。
    """
    with open(features_path, 'rb') as f:
        features = pickle.load(f)

    print(f"特征已从 {features_path} 加载")
    return features


def compute_accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (jt.Var): prediction matrix with shape (batch_size, num_classes).
        target (jt.Var): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred == target.view(1, -1).expand_as(pred)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc = correct_k * (100.0 / batch_size)
        res.append(acc.item())

    return res