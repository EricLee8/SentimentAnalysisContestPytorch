import torch
import numpy as np


def compute_one_f1(pred, target, label_class):
    TP = ((pred == label_class) & (target == label_class)).cpu().sum().item() # 预测为label_class且正确的样本数
    all_pred = (pred == label_class).cpu().sum().item() # 所有预测为label_class的样本数
    all_target = (target == label_class).cpu().sum().item() # 所有标签为label_class的样本数
    precision = TP/all_pred if all_pred != 0 else 0. # 精确率
    recall = TP/all_target if all_target != 0 else 0. # 召回率
    f1 = 2*recall*precision / (recall+precision) if (recall+precision) != 0 else 0.
    return f1


def compute_f1(pred, target):
    f1_0 = compute_one_f1(pred, target, 0)
    f1_1 = compute_one_f1(pred, target, 1)
    f1_2 = compute_one_f1(pred, target, 2)
    f1 = (f1_0 + f1_1 + f1_2) / 3
    return f1

    
    