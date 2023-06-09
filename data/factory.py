import utils.torch as ptu
import numpy as np

from data.coco import COCODataset
from data.loader import Loader
from torch.utils.data import SubsetRandomSampler


def create_dataset(dataset_kwargs):
    dataset_kwargs = dataset_kwargs.copy()
    dataset_name = dataset_kwargs.pop("dataset")
    batch_size = dataset_kwargs.pop("batch_size")
    num_workers = dataset_kwargs.pop("num_workers")
    split = dataset_kwargs.pop("split")

    if dataset_name == 'coco':
        dataset = COCODataset(split=split, **dataset_kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} is unknown.")

    # validation_split = 0.95
    # shuffle_dataset = True
    # random_seed = 42
    #
    # # Creating data indices for training and validation splits:
    # dataset_size = len(dataset)
    # indices = list(range(dataset_size))
    # split = int(np.floor(validation_split * dataset_size))
    # if shuffle_dataset:
    #     np.random.seed(random_seed)
    #     np.random.shuffle(indices)
    # train_indices, val_indices = indices[split:], indices[:split]
    #
    # # Creating PT data samplers and loaders:
    # train_sampler = SubsetRandomSampler(train_indices)
    # valid_sampler = SubsetRandomSampler(val_indices)

    # dataset = Loader(
    #     dataset=dataset,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     distributed=ptu.distributed,
    #     split=split,
    #     train_sampler=train_sampler,
    # )
    dataset = Loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=ptu.distributed,
        split=split,
    )
    return dataset
