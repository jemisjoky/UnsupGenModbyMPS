#!/usr/bin/env python3
import gzip
import pickle
from math import sqrt
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from MPScumulant_torch import MPS_c


@torch.no_grad()
def sample(cores, edge_vecs, num_samples=1, embed_fun=None):
    """
    Produce continuous or discrete samples from an MPS Born machine

    Args:
        cores: Collection of n core tensors, for n the number of pixels in the
            input to the MPS.
        edge_vecs: Pair of vectors giving the left and right boundary
            conditions of the MPS.
        num_samples: Number of samples to generate in parallel.
        embed_fun: Embedding function used to sample continuous inputs

    Returns:
        samples: Tensor of shape (num_samples, n) containing the sample values.
            These will be nonnegative integers, or floats if the parent MPS is
            using an embedding of a continuous domain.
    """
    input_dim = cores[0].shape[0]
    assert len(edge_vecs) == 2
    left_vec, right_vec = edge_vecs

    # Get right PSD matrices resulting from tracing over right cores of MPS
    right_mats = right_trace_mats(cores, right_vec)
    assert len(right_mats) == len(cores)

    # Precompute cumulative embedding mats for non-trivial embedding
    if embed_fun is not None:
        num_points = 1000
        points = torch.linspace(0.0, 1.0, steps=num_points)
        dx = 1 / (num_points - 1)
        emb_vecs = torch.tensor(embed_fun(points))
        assert emb_vecs.shape[1] == input_dim

        # Get rank-1 matrices for each point, then numerically integrate
        emb_mats = torch.einsum("bi,bj->bij", emb_vecs, emb_vecs.conj())
        int_mats = torch.cumsum(emb_mats, dim=0) * dx

    else:
        num_points = input_dim
        int_mats = None
        points = None

    # Initialize conditional left PSD matrix and generate samples sequentially
    l_vecs = left_vec[None].expand(num_samples, -1)
    samples, step = [], 0
    for core, r_mat in zip(cores, right_mats):
        print(f"\rSample {step}", end="")
        step += 1
        samps, l_vecs = _sample_step(
            core,
            l_vecs,
            r_mat,
            embed_fun,
            int_mats,
            num_samples,
            points,
        )
        samples.append(samps)
    print()
    samples = torch.stack(samples, dim=1)

    # If needed, convert integer sample outcomes into continuous values
    if embed_fun is not None:
        samples = points[samples]
    elif input_dim != 2:
        samples = samples.float() / (input_dim - 1)

    return samples


@torch.no_grad()
def _sample_step(
    core,
    l_vecs,
    r_mat,
    embed_fun,
    int_mats,
    num_samples,
    points,
):
    """
    Function for generating single batch of samples
    """
    # Get unnormalized probabilities and normalize
    if embed_fun is not None:
        int_probs = torch.einsum(
            "bl,bm,ilr,uij,jms,rs->bu",
            l_vecs,
            l_vecs.conj(),
            core,
            int_mats,
            core.conj(),
            r_mat,
        )
        int_probs /= int_probs[:, -1][:, None]
    else:
        probs = torch.einsum(
            "bl,bm,ilr,ims,rs->bi", l_vecs, l_vecs.conj(), core, core.conj(), r_mat
        )
        probs /= probs.sum(dim=1, keepdim=True)
        int_probs = torch.cumsum(probs, axis=1)

    if int_probs.is_complex():
        int_probs = int_probs.real
    try:
        assert torch.all(int_probs >= 0)  # Tolerance for small negative values
    except AssertionError:
        print(int_probs)
        print("WARNING: Some negative probabilities found")
    assert torch.allclose(int_probs[:, -1], torch.ones(1))

    # Sample from int_probs (argmax finds first int_p with int_p > rand_val)
    # rand_vals = torch.rand((num_samples, 1), generator=generator)
    rand_vals = torch.rand((num_samples, 1))
    samp_ints = torch.argmax((int_probs > rand_vals).long(), dim=1)

    # Conditionally update new left boundary vectors
    if embed_fun is not None:
        samp_points = points[samp_ints]
        emb_vecs = torch.tensor(embed_fun(samp_points))
        l_vecs = torch.einsum("bl,ilr,bi->br", l_vecs, core, emb_vecs)
    else:
        samp_mats = core[samp_ints]
        l_vecs = torch.einsum("bl,blr->br", l_vecs, samp_mats)
    # Rescale all vectors to have unit 2-norm
    l_vecs /= torch.norm(l_vecs, dim=1, keepdim=True)

    return samp_ints, l_vecs


@torch.no_grad()
def right_trace_mats(tensor_cores, right_vec):
    """
    Generate virtual PSD matrices from tracing over right cores of MPS

    Note that resultant PSD matrices are rescaled to avoid exponentially
    growing or shrinking trace.

    Args:
        tensor_cores: Collection of n core tensors, for n the number of pixels
            in the input to the MPS.
        right_vec: The vector giving the right boundary condition of the MPS.

    Returns:
        right_mats: Collection of n PSD matrices, ordered from left to right.
    """
    uniform_input = hasattr(tensor_cores, "shape")
    assert not uniform_input or tensor_cores.ndim == 4

    # Build up right matrices iteratively, from right to left
    r_mat = right_vec[:, None] @ right_vec[None].conj()
    right_mats, step = [r_mat], len(tensor_cores)
    for core in tensor_cores.flip(dims=[0])[:-1]:  # Iterate backwards up to first core
        print(f"\rMarginalize {step}", end="")
        step -= 1
        r_mat = torch.einsum("ilr,ims,rs->lm", core, core.conj(), r_mat)
        # Stabilize norm
        r_mat /= torch.trace(r_mat)
        right_mats.append(r_mat)
    print()

    if uniform_input:
        right_mats = torch.stack(right_mats[::-1])

    return right_mats


def print_pretrained_samples():
    # Extract core tensors and edge vectors from saved model, convert to torch
    num_samps = 10
    # Discrete MPS models
    # save_file = "MNIST/rand1k_runs/mnist1k_27/mps_loop_050.model.gz"    # BD02
    # save_file = "MNIST/rand1k_runs/mnist1k_31/mps_loop_050.model.gz"    # BD10
    # save_file = "MNIST/rand1k_runs/mnist1k_23/mps_loop_050.model.gz"    # BD20
    # save_file = "MNIST/rand1k_runs/mnist1k_25/mps_loop_050.model.gz"    # BD40
    # save_file = "MNIST/rand1k_runs/mnist1k_40/mps_loop_020.model.gz"    # BD70
    # save_file = "MNIST/rand1k_runs/mnist1k_41/mps_loop_020.model.gz"    # BD100
    # save_file = "MNIST/rand1k_runs/mnist1k_42/mps_loop_015.model.gz"    # BD150

    # Continuous MPS models
    input_path = "log_dir/experiment_020"
    # save_file = "MNIST/rand1k_runs/mnist1k_44/mps_loop_016.model.gz"    # BD10
    # save_file = "MNIST/rand1k_runs/mnist1k_46/mps_loop_016.model.gz"    # BD20
    # save_file = "MNIST/rand1k_runs/mnist1k_47/mps_loop_010.model.gz"    # BD30
    # save_file = "MNIST/rand1k_runs/mnist1k_48/mps_loop_012.model.gz"    # BD40
    # save_file = "MNIST/rand1k_runs/mnist1k_49/mps_loop_014.model.gz"    # BD50

    # Get a list of all the model savefiles we're sampling from
    input_path = Path(input_path).absolute()
    assert input_path.exists()
    print_str = f"Producing {num_samps} samples from"
    if input_path.is_file():
        save_dir = input_path.parent
        model_list = [input_path]
        print_str += f": {input_path}"
    else:
        save_dir = input_path
        model_list = [f for f in input_path.iterdir() if f.suffix == ".model"]
        print_str += f" each of: {', '.join([str(f) for f in model_list])}"
    print(print_str)

    for save_file in model_list:
        with gzip.open(save_file, "rb") as f:
            mps = pickle.load(f)
        core_tensors, edge_vecs = MPS_c.export_params(mps)
        if not isinstance(core_tensors, torch.Tensor):
            core_tensors = torch.tensor(core_tensors, dtype=torch.float32)
            edge_vecs = torch.tensor(edge_vecs, dtype=torch.float32)
        else:
            core_tensors = core_tensors.to(torch.float32)
            edge_vecs = edge_vecs.to(torch.float32)

        samples = sample(
            core_tensors, edge_vecs, num_samples=num_samps, embed_fun=mps.embed_fun
        )

        # Reshape and rescale sampled values
        width = round(sqrt(core_tensors.shape[0]))
        samples = samples.reshape(num_samps, width, width)
        assert torch.all(samples >= 0)
        # if torch.any(samples > 1):
        #     samples = samples.float() / samples.max()

        # Plot everything with new sampler
        for i, im in enumerate(samples, start=1):
            # Normalize image
            cmap, norm = plt.cm.gray_r, plt.Normalize()
            image = cmap(norm(im))

            # Create image save path
            im_path = save_dir / f"{save_file.stem}_{i:02d}.png"
            plt.figure(figsize=(5, 5))
            plt.imshow(image, interpolation="nearest")
            plt.savefig(im_path)


if __name__ == "__main__":
    print_pretrained_samples()
