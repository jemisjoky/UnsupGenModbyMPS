import os

import torch
import numpy as np


T_INTS = tuple(getattr(torch, f"int{nb}") for nb in [8, 16, 32, 64])
NP_INTS = tuple(getattr(np, f"int{nb}") for nb in [8, 16, 32, 64]) + (np.uint8,)
INTS = NP_INTS + T_INTS
T_FLOATS = tuple(getattr(torch, f"float{nb}") for nb in [16, 32, 64])
NP_FLOATS = tuple(getattr(np, f"float{nb}") for nb in [16, 32, 64, 128])
FLOATS = NP_FLOATS + T_FLOATS
COMPLEXES = tuple(getattr(np, f"complex{nb}") for nb in [64, 128, 256])


def rm_intermediate_checkpoints(exp_dir="./MNIST/rand1k_runs/"):
    """
    Delete all experiment checkpoints except the final one
    """
    for folder in os.listdir(exp_dir):
        folder = f"{exp_dir}{folder}/"
        file_list = sorted(os.listdir(folder))
        for file in file_list[:-1]:
            if file.endswith(".json"):
                continue
            os.remove(folder + file)


def onehot(data, num_bins):
    """
    Convert discrete input data into one-hot encoded vectors
    """
    assert is_int_type(data)
    shape = data.shape
    numel = data.size
    # Create flattened version of output, reshape before returning
    output = np.zeros((numel, num_bins))
    output[np.arange(numel), data.reshape(-1)] = 1
    return output.reshape(shape + (num_bins,))


def is_int_type(array):
    """
    Computes whether input array is integral type (e.g. np.intXX)
    """
    return array.dtype in INTS


def is_float_type(array):
    """
    Computes whether input array is non-complex float type (e.g. np.floatXX)
    """
    return array.dtype in FLOATS


def is_complex_type(array):
    """
    Computes whether input array is complex type (e.g. np.complexXX)
    """
    assert isinstance(array, np.ndarray)
    return array.dtype in COMPLEXES


if __name__ == "__main__":
    from sys import argv

    assert len(argv) > 1

    if argv[1] == "rm_intermediate":
        rm_intermediate_checkpoints()
