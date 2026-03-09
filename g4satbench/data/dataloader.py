import torch
import numpy as np
import random
import itertools

from g4satbench.data.dataset import SATDataset
from torch_geometric.data import Batch
from torch.utils.data import DataLoader


def collate_fn(batch):
    return_batch = dict()
    keys = list(batch[0].keys())
    length = len(batch)
    for key in keys:
        return_batch[key] = Batch.from_data_list(
            [s for s in list(itertools.chain(*[batch[idx][key] for idx in range(length)]))])
    return return_batch
    # return Batch.from_data_list([s for s in list(itertools.chain(*batch))])


def get_dataloader(data_dir, splits, sample_size, problem_types, opts, mode, use_contrastive_learning=False):
    dataset = SATDataset(data_dir, splits, sample_size, use_contrastive_learning, mode, problem_types, opts)
    batch_size = opts.batch_size // len(splits) if opts.data_fetching == 'parallel' else opts.batch_size

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode=='train'),
        collate_fn=collate_fn,
        pin_memory=True,
    )
