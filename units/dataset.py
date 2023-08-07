import bisect
import math
import random
import warnings

import numpy as np
import torch
from torch.utils import data

from units.structures import Batch


class SampleError(Exception):
    """Raised when sample is malformed"""


class MultitaskDataset(data.Dataset):
    """General Multi-task Dataset
    Args:
        datasource: Datasource class form datasource.py
        mappers: List of mappers that generates the input for the model
        transform: dataset-wise transform object.
            Image Tensorize operation should be contained.
    """

    def __init__(self, datasource, mappers, transform):
        super().__init__()
        self.datasource = datasource
        self.mappers = mappers
        self.transform = transform

    def __len__(self):
        return len(self.datasource)

    def __getitem__(self, index):
        """
        Args:
            index (int): data index

        Returns:
            image (Tensor[float32]): tensorized image (C, H, W)
            sample (structures.Sample): container that has the input for the model
        """
        img, sample = self.datasource[index]

        if self.transform is not None:
            img, sample = self.transform(img, sample)

        _, height, width = img.shape
        sample.image_size = (height, width)

        # We expect complaints about the samples only occurs in the mapper
        # But maybe it is better to capture entire __getitem__?
        try:
            for mapper in self.mappers:
                input = mapper(sample)
                sample.set(mapper.name, input)

        except SampleError:
            try:
                warnings.warn(
                    f"Failed to use {index} of {self.datasource}; try {index + 1}"
                )

                return self.__getitem__(index + 1)

            except IndexError:
                rand_index = random.randrange(len(self))

                warnings.warn(
                    f"Failed to use {index + 1} of {self.datasource}; try {rand_index}"
                )

                return self.__getitem__(rand_index)

        return img, sample


class MultitaskCollator:
    """General Multi-task Collator
    1. Get (max_h, max_w) from batch.
    2. Use img_multiple to get canvas which has multiply of img_multiple.
        (i.e. canvas % 32 == 0)

    This collate funtion maintains aspect ratio of images with padding.
    img_multiple is useful when you concat multiple features using FPN.

    Args:
        mappers: List of mappers that generates the input for the model
        img_multiple (Union[int, float]): img-size multiplication

    Returns:
        batch (structures.Batch): container that contains:
            images (Tensor[float32]): (B, C, H, W)
            masks (Tensor[bool]): (B, H, W)
            and additional inputs that mapper produces for the model
    """

    def __init__(self, mappers, evaluate=False, img_multiple=32):
        self.mappers = mappers
        self.evaluate = evaluate
        self.img_multiple = img_multiple

    def __call__(self, batch):
        img_shapes = [img.shape for img, _ in batch]
        max_h = max([s[1] for s in img_shapes])
        max_w = max([s[2] for s in img_shapes])

        height = math.ceil(max_h / self.img_multiple) * self.img_multiple
        width = math.ceil(max_w / self.img_multiple) * self.img_multiple
        b_size = len(batch)

        images = torch.zeros(
            b_size, batch[0][0].shape[0], height, width, dtype=torch.float32
        )
        masks = torch.ones(b_size, height, width, dtype=torch.bool)

        for i, (img, _) in enumerate(batch):
            _, height, width = img.shape
            images[i, :, :height, :width] = img
            masks[i, :height, :width] = False

        return_batch = Batch(images=images, masks=masks)
        samples = [sample for _, sample in batch]

        for mapper in self.mappers:
            input = mapper.collate_fn(samples)
            return_batch.set(mapper.name, input)

        if self.evaluate:
            return_batch.set("samples", samples)

        return return_batch


class WeightedDataset(data.Dataset):
    def __init__(self, datasets, ratios, names=None):
        super().__init__()
        self.ratios = ratios
        sizes = np.array([len(d) for d in datasets])
        ratios = np.asarray(ratios)
        self.names = names
        n_sample = sizes.sum() * ratios
        target_sample = n_sample / np.min(n_sample / sizes)
        self.datasets = datasets
        self.target_sample = np.round(target_sample).astype(np.int)
        self.points = np.cumsum(self.target_sample).tolist()

    def summary(self):
        for i, (dset, ratio, sample) in enumerate(
            zip(self.datasets, self.ratios, self.target_sample.tolist())
        ):
            if self.names is not None:
                print(
                    f"#{i} {self.names[i]} total: {len(dset)} ratio: {ratio} sample: {sample}"
                )

            else:
                print(f"#{i} total: {len(dset)} ratio: {ratio} sample: {sample}")

    def __len__(self):
        return self.points[-1]

    def __getitem__(self, index):
        point = bisect.bisect(self.points, index)
        dataset = self.datasets[point]

        if point == 0:
            diff = index

        else:
            diff = index - self.points[point - 1]

        return dataset[diff % len(dataset)]
