import torch
from functools import partial
from torch.utils.data.dataloader import default_collate


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def seg_collate(batch):
    elem = batch[0]

    if isinstance(elem, dict):
        return batch
    elif isinstance(elem, tuple):
        return [seg_collate(samples) for samples in zip(*batch)]
    else:
        return default_collate(batch)


def toy_collate(batch):
    images, targets = list(zip(*batch))
    images = torch.stack(images, 0)
    return images, targets
