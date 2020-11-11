__author__ = 'max'

import torch
import torch.nn as nn
from torch._six import inf


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def clip_grad_norm_(parameters, max_norm, norm_type=2, mode='total'):
    assert max_norm > 0
    if mode == 'total':
        return nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)
    elif mode == 'each':
        return clip_param_grad_norm_(parameters, max_norm, norm_type=norm_type)
    else:
        raise ValueError('Unknown grad clip mode {}.'.format(mode))


def clip_param_grad_norm_(parameters, max_norm, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    grads = [p.grad.detach() for p in parameters]

    grad_norms = []
    for grad in grads:
        if norm_type == inf:
            grad_norm = grad.abs().max()
        else:
            grad_norm = torch.norm(grad, p=norm_type, dtype=torch.float32)
        grad_norms.append(grad_norm)
        clip_coef = max_norm / (grad_norm + 1e-6)
        if clip_coef < 1:
            grad.mul_(clip_coef)

    return torch.max(torch.stack(grad_norms))
