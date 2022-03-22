"""Dataset loaders for MNIST, fashion MNIST, and Genz time series"""
import os

import numpy as np
import torch
import torchvision
from torchvision import transforms as tf


@torch.no_grad()
def bin_data(input, num_bins=None):
    """
    Discretize greyscale values into a finite number of bins
    """
    if num_bins is None:
        return input
    assert num_bins > 0

    # Set each of the corresponding bin indices
    out_data = torch.full_like(input, -1)
    for i in range(num_bins):
        bin_inds = (i / num_bins <= input) * (input <= (i + 1) / num_bins)
        out_data[bin_inds] = i
    assert out_data.max() >= 0

    return out_data.long()


def load_genz(genz_num: int, slice_len=None):
    """
    Load a dataset of time series with dynamics set by various Genz functions

    Separate train, validation, and test datasets are returned, containing data
    from 8000, 1000, and 1000 time series. The length of each time series
    depends on `slice_len`, and is 100 by default (`slice_len=None`). For
    positive integer values of `slice_len`, these time series are split into
    contiguous chunks of length equal to `slice_len`.

    Args:
        genz_num: Integer between 1 and 6 setting choice of Genz function

    Returns:
        train, val, test: Three arrays with respective shape (8000, 100, 1),
            (1000, 100, 1), and (1000, 100, 1).
    """
    # Length between startpoints of output sliced series
    stride = 2
    assert 1 <= genz_num <= 6
    assert slice_len is None or 1 < slice_len <= 100
    if slice_len is None:
        slice_suffix = ""
        slice_len = 100
    else:
        assert isinstance(slice_len, int)
        slice_suffix = f"_l{slice_len}_s{stride}"

    # Number of slices per time series
    s_per_ts = (100 - slice_len) // stride + 1

    # Return saved dataset if we have already generated this previously
    save_file = f"datasets/genz/genz{genz_num}{slice_suffix}.npz"
    if os.path.isfile(save_file):
        out = np.load(save_file)
        train, val, test = out["train"], out["val"], out["test"]
        assert val.shape == test.shape == (1000 * s_per_ts, slice_len)
        assert train.shape == (8000 * s_per_ts, slice_len)
        return train, val, test

    # Definitions of each of the Genz functions which drive the time series
    gfun = genz_funs[genz_num]

    # Initialize random starting values and update using Genz update function
    rng = np.random.default_rng(genz_num)
    x = rng.permutation(np.linspace(0.0, 1.0, num=10000))
    long_series = np.empty((10000, 100))
    for i in range(100):
        x = gfun(x)
        long_series[:, i] = x

    # Normalize the time series values to lie in range [0, 1]
    min_val, max_val = long_series.min(), long_series.max()
    long_series = (long_series - min_val) / (max_val - min_val)

    # Split into train, validation, and test sets
    base_series = (long_series[:8000], long_series[8000:9000], long_series[9000:])

    # Cut up the full time series into shorter sliced time series
    all_series = []
    for split in base_series:
        num_series = split.shape[0]
        s_split = np.empty((num_series * s_per_ts, slice_len))
        for i in range(s_per_ts):
            j = i * stride
            s_split[i*num_series:(i+1)*num_series] = split[:, j:(j + slice_len)]
        all_series.append(s_split)

    # Shuffle individual time series, save everything to disk
    train, val, test = [rng.permutation(ts) for ts in all_series]
    np.savez_compressed(save_file, train=train, val=val, test=test)
    return train, val, test

w = 0.5
c = 1.0  # I'm using the fact that c=1.0 to set c**2 = c**-2 = c
genz_funs = [
    None,  # Placeholder to give 1-based indexing
    lambda x: np.cos(2 * np.pi * w + c * x),
    lambda x: (c + (x + w)) ** -1,
    lambda x: (1 + c * x) ** -2,
    lambda x: np.exp(-c * np.pi * (x - w) ** 2),
    lambda x: np.exp(-c * np.pi * np.abs(x - w)),
    lambda x: np.where(x > w, 0, np.exp(c * x)),
]
