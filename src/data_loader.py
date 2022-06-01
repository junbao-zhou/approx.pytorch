import glob
import os
import pathlib
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset
from util import Statistic

current_file_path = pathlib.Path(__file__).parent.resolve()
dataset_dir = f'{current_file_path}/../data'
IS_DOWNLOAD_DATA = False


class DatasetStat():
    def __init__(self, mean: float, std: float, var: float) -> None:
        self.mean = mean
        self.std = std
        self.var = var


DATASET_METAS = {
    'FashionMNIST': DatasetStat(0.28614094853401184, 0.35289040207862854, 0.12453246116638184),
    'MNIST': DatasetStat(0.13129360973834991, 0.30881890654563904, 0.0953928753733635),
    'CIFAR10': DatasetStat([0.4919, 0.4827, 0.4472], [0.2469, 0.2434, 0.2616], [0.0610, 0.0592, 0.0684]),
    'CIFAR100': DatasetStat([0.5074, 0.4867, 0.4411], [0.2675, 0.2566, 0.2763], [0.0715, 0.0658, 0.0763]),
    'ImageNet': DatasetStat([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], [0.0745, 0.0698, 0.0768]),
}


def data_set(dataset_name: str, is_normalize: bool = True, is_augment: bool = False, image_net_out_size: int = 227):
    dataset = eval('datasets.' + dataset_name)

    basic_transform = transforms.Compose(
        ([transforms.Resize(256),
          transforms.CenterCrop(256),
          transforms.RandomCrop(image_net_out_size)]
            if dataset_name == 'ImageNet' else []) +
        [transforms.ToTensor()] +
        ([transforms.Normalize(
            DATASET_METAS[dataset_name].mean,
            DATASET_METAS[dataset_name].std)]
            if is_normalize else [])
    )

    if dataset_name == 'ImageNet':
        train_data = datasets.ImageNet(
            root=dataset_dir,
            split='train',
            transform=basic_transform,
        )
    else:
        train_data = dataset(
            root=dataset_dir,
            train=True,
            transform=basic_transform,
            download=IS_DOWNLOAD_DATA
        )
    if is_augment:
        origin_size = train_data[0][0].shape[-1]
        train_data.transform = transforms.Compose([
            transforms.Pad(int(origin_size / 2), padding_mode='symmetric'),
            transforms.RandomRotation(11),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(int(origin_size * 1.2)),
            transforms.RandomCrop(origin_size),
            transforms.ToTensor(),
        ] + (
            [transforms.Normalize(
                DATASET_METAS[dataset_name].mean,
                DATASET_METAS[dataset_name].std)] if is_normalize else []
        ))
    print(train_data)
    if dataset_name == 'ImageNet':
        test_data = datasets.ImageNet(
            root=dataset_dir,
            split='val',
            transform=transforms.Compose(
                [transforms.Resize(256),
                 transforms.CenterCrop(image_net_out_size),
                 transforms.ToTensor(), ] +
                ([transforms.Normalize(
                    DATASET_METAS[dataset_name].mean,
                    DATASET_METAS[dataset_name].std)]
                    if is_normalize else [])
            ),
        )
    else:
        test_data = dataset(
            root=dataset_dir,
            train=False,
            transform=basic_transform,
            download=IS_DOWNLOAD_DATA
        )
    print(test_data)

    return train_data, test_data


def data_loader(dataset_name: str, batch_size: int, is_normalize: bool = True, is_augment: bool = False, image_net_out_size: int = 227, is_shuffle_valid_loader = False):
    train_data, val_data = data_set(
        dataset_name, is_normalize, is_augment, image_net_out_size)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True)

    valid_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=is_shuffle_valid_loader)

    return train_loader, valid_loader


def dataset_statistic(dataset_name: str, batch_size: int, is_augment: bool = False):
    train_loader, val_loader = data_loader(
        dataset_name, batch_size, is_normalize=False, is_augment=is_augment)
    stats_list = []
    for loader in (train_loader, val_loader):
        for X, label in loader:
            print(X.shape)
            stat = Statistic(X, dataset_name, dim=[0, 2, 3], is_fig=True)
            print(f'batch {X.size(0)}:{stat}')
            stats_list.append(stat)
            print(Statistic.merge(stats_list))


if __name__ == '__main__':
    # train_loader, valid_loader = data_loader('FashionMNIST', 500)
    # inputs, classes = next(iter(train_loader))
    # print(inputs.shape)
    # print(inputs.min(), inputs.max())
    dataset_statistic('CIFAR100', 2000, is_augment=False)
    # transforms.ToPILImage()(x).save("test.png")
