import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler

from common.io import info


def get_train_validation_loaders1(dataset):
    def __get_data_loader(indices):
        sampler = SubsetRandomSampler(indices)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=dataset.batch_size,
                                                  sampler=sampler)
        return data_loader

    train_dataset = dataset
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(dataset.validation_split * dataset_size))
    if dataset.shuffle_dataset:
        np.random.seed(dataset.random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # info(f"train size={len(train_indices)} val size={len(val_indices)}")

    train_loader = __get_data_loader(train_indices)
    validation_loader = __get_data_loader(val_indices)
    return train_loader, validation_loader


def get_data_loader(dataset, batch_size, data_processing=None):
    # return torch.utils.data.DataLoader(dataset,
    #                                    batch_size=batch_size,
    #                                    shuffle=True,
    #                                    collate_fn=lambda x: list(zip(*x)))
    if data_processing is None:
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    kwargs = {}
    return torch.utils.data.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       collate_fn=lambda x: data_processing(x),
                                       **kwargs)


def get_train_validation_loaders(dataset, batch_size=32, seed_num=43):
    # torch.manual_seed(seed_num)
    # split_size = int(np.floor(0.1 * len(dataset)))
    # info(f"dataset_size={len(dataset)} split_size={split_size} batch_size={batch_size}")
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset_train = torch.utils.data.Subset(dataset, indices[:-split_size])
    # dataset_validation = torch.utils.data.Subset(dataset, indices[-split_size:])

    dataset_train, dataset_validation = get_train_validation_datasets(dataset)
    data_loader_train = get_data_loader(dataset_train, batch_size)
    data_loader_validation = get_data_loader(dataset_validation, batch_size)

    return data_loader_train, data_loader_validation


def get_train_validation_datasets(dataset, batch_size=32, seed_num=43):
    torch.manual_seed(seed_num)
    split_size = int(np.floor(0.1 * len(dataset)))
    info(f"dataset_size={len(dataset)} split_size={split_size} batch_size={batch_size}")
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices[:-split_size])
    dataset_validation = torch.utils.data.Subset(dataset, indices[-split_size:])
    return dataset_train, dataset_validation
