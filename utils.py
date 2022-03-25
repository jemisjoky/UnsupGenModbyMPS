import os
from math import log, log2

import torch
import numpy as np


T_INTS = tuple(getattr(torch, f"int{nb}") for nb in [8, 16, 32, 64])
NP_INTS = tuple(getattr(np, f"int{nb}") for nb in [8, 16, 32, 64]) + (np.uint8,)
T_FLOATS = tuple(getattr(torch, f"float{nb}") for nb in [16, 32, 64])
NP_FLOATS = tuple(getattr(np, f"float{nb}") for nb in [16, 32, 64, 128])
COMPLEXES = tuple(getattr(np, f"complex{nb}") for nb in [64, 128, 256])


def init_stable(tensor):
    """
    Initialize log register for an input tensor
    """
    return tensor.new_zeros((), dtype=torch.int64)


def stabilize(tensor, log_register):
    """
    Rescale input tensor by power of 2, with rescaling added to log register
    """
    # Compute a power-of-two rescaling factor for tensor
    log2_norm = tensor.abs().sum().log2()
    log2_target = log2(tensor.numel())
    log2_rescale = (log2_target - log2_norm).long()

    # Ignore rescale if it is infinite (input tensor is zero tensor)
    if ~log2_rescale.isfinite():
        log2_rescale[()] = 0

    # Incorporate rescaling in tensor and log register
    return tensor * 2.0**log2_rescale, log_register - log2_rescale


def destabilize(tensor, log_register):
    """
    Incorporate log register rescaling into input tensor
    """
    return tensor * 2.0**log_register


def stable_log(tensor, log_register):
    """
    Compute natural log of input tensor, incorporating log register
    """
    return tensor.log() + log(2) * log_register


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
    assert isinstance(array, (np.ndarray, torch.Tensor))
    if isinstance(array, np.ndarray):
        return array.dtype in NP_INTS
    else:
        return array.dtype in T_INTS


def is_float_type(array):
    """
    Computes whether input array is non-complex float type (e.g. np.floatXX)
    """
    assert isinstance(array, (np.ndarray, torch.Tensor))
    if isinstance(array, np.ndarray):
        return array.dtype in NP_FLOATS
    else:
        return array.dtype in T_FLOATS


def is_complex_type(array):
    """
    Computes whether input array is complex type (e.g. np.complexXX)
    """
    assert isinstance(array, np.ndarray)
    return array.dtype in COMPLEXES


if __name__ == "__main__":
    # Test out stabilization code
    for n in list(range(100)) + [torch.zeros(())]:
        if isinstance(n, int):
            order = n // 20
            shape = tuple(range(2, 2 + order))
            orig_tensor = torch.randn(shape) * (10 ** (16 * torch.rand(()) - 8))
        else:
            orig_tensor = n

        assert torch.allclose(
            destabilize(*stabilize(orig_tensor, init_stable(orig_tensor))),
            orig_tensor,
        )
        abs_tensor = orig_tensor.abs()
        assert torch.allclose(
            stable_log(*stabilize(abs_tensor, init_stable(abs_tensor))),
            abs_tensor.log()
            )
