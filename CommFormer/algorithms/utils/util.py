import copy
import numpy as np

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    """初始化网络模块权重

    Args:
        module (nn.Module): 被初始化的模块
        weight_init (function): 权重初始化函数，例如nn.init.orthogonal_（正交矩阵初始化权重）
        bias_init (function): 偏差初始化函数，
        gain (int, optional): 缩放因子，用于调整初始化的方差。通常对于 ReLU 激活函数设置为 根号2，对于线性激活函数为 1。

    Returns:
        nn.Module: 完成初始化的模块，即第一个输入参数
    """
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output