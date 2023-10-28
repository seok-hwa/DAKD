import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def cross_entropy_2d(predict, target, weight, cfg):
    """
    Args:
        predict:(n, c, h, w)
        target:(n, h, w)
    """
    assert not target.requires_grad
    assert predict.dim() == 4
    assert target.dim() == 3
    assert predict.size(0) == target.size(0), f"{predict.size(0)} vs {target.size(0)}"
    assert predict.size(2) == target.size(1), f"{predict.size(2)} vs {target.size(1)}"
    assert predict.size(3) == target.size(2), f"{predict.size(3)} vs {target.size(3)}"
    
    n, c, h, w = predict.size()
    target_mask = (target >= 0) * (target != cfg.INPUT.IGNORE_LABEL)
    target = target[target_mask]
    if not target.data.dim():
        return Variable(torch.zeros(1))
    
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    
    loss = F.cross_entropy(predict, target, reduce=False)
    loss = torch.mean(loss * weight[target_mask].view(-1))
    return loss