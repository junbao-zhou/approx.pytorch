from pyexpat import features
from time import time
import torch
from typing import Iterable, List, Union
import matplotlib.pyplot as plt
import numpy as np


def exec_time(func):
    start = time()
    res = func()
    return (time() - start, res)


def print_var(var):
    print(f'{var.__name__} : {var}')


class Statistic():
    features_str = ['mean', 'std', 'var', 'min', 'max']

    def __init__(self, tensor: torch.Tensor, name: str, dim=None, is_fig: bool = False, bins=128, range=None) -> None:
        self.name = name
        if type(tensor) != torch.Tensor:
            tensor = torch.Tensor(tensor)
        for f_str in Statistic.features_str:
            if (f_str not in ['min', 'max']) and dim != None:
                self.__setattr__(f_str, tensor.__getattribute__(f_str)(dim))
            else:
                self.__setattr__(f_str, tensor.__getattribute__(f_str)())
        self.zero_rate = 1 - (tensor.count_nonzero() / tensor.nelement())

        if is_fig:
            plt.hist(tensor.flatten().numpy(), bins=bins, range=range)
            plt.savefig(f'{self.name}.png')
            plt.show()
            plt.close()

    def __repr__(self) -> str:
        res = [f'{f_str} = {self.__getattribute__(f_str)}'for f_str in Statistic.features_str]
        res = '\n'.join(res)
        return f"""{self.name}
{res}
zero_rate  = {self.zero_rate.item()}
"""

    def merge(stat_list: List):
        total_stats = {}
        for f_str in Statistic.features_str:
            total_stats[f_str] = []
        for stat in stat_list:
            for f_str in Statistic.features_str:
                total_stats[f_str].append(stat.__getattribute__(f_str))
        
        for s in total_stats:
            total_stats[s] = torch.stack(total_stats[s])
        stat = Statistic([0], 'merged')
        stat.mean = total_stats['mean'].mean(dim=0)
        stat.std = total_stats['std'].mean(dim=0)
        stat.var = total_stats['var'].mean(dim=0)
        stat.min = total_stats['min'].min()
        stat.max = total_stats['max'].max()
        return stat


class NumCollect():
    def __init__(self) -> None:
        self.nums = []

    def add(self, num):
        self.nums.append(num)

    def stat(self):
        return Statistic(self.nums)


def compare(a, b, name: str):
    print(f'==== comparing {name} ====')
    is_equal = a.equal(b)
    print(f'is_equal = {is_equal}')
    if not is_equal:
        # print(a)
        # print(b)
        error = torch.abs(a - b)
        # print(f'error = {error}')
        max_error = error.max()
        print(f'max_error = {max_error}')
        total_error = error.sum()
        print(f'total_error = {total_error}')
        error_num = (error != 0).sum()
        print(f'error_num = {error_num}')
        print(f'total_len = {a.nelement()}')
        avg_error = float(total_error) / float(a.nelement())
        print(f'avg_error = {avg_error}')


def expand(x, n: int):
    if isinstance(x, Iterable):
        return tuple(x)
    return tuple([x]*n)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

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

    def __str__(self):
        fmtstr = '{name} : {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

    def __repr__(self) -> str:
        return self.__str__()