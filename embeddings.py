# MIT License
#
# Copyright (c) 2021 Jacob Miller
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Uniform and non-uniform probabilistic MPS classes"""
from math import sqrt
from functools import partial

import torch
import numpy as np
from torch import einsum


class FixedEmbedding:
    r"""
    Framework for fixed embedding function converting data to vectors

    Args:
        emb_fun (function): Function taking arbitrary tensors of values and
            returning tensor of embedded vectors, which has one additional
            axis in the last position. These values must be either integers,
            for discrete data domains, or reals, for continuous data domains.
        data_domain (DataDomain): Object which specifies the domain on which
            the data fed to the embedding function is defined.
        frameify (bool): Whether to rescale embedding vectors to make the
            embedding function into a frame. (Default: False)
        dtype: Datatype to save associated lambda matrix in. (Default: float32)
    """

    def __init__(
        self,
        emb_fun,
        min_domain=0.0,
        max_domain=1.0,
        num_points=1000,
        frameify=False,
    ):
        assert hasattr(emb_fun, "__call__")
        assert min_domain < max_domain

        # Save defining data
        self.emb_fun = emb_fun
        self.domain = (min_domain, max_domain)
        self.frameify = frameify

        # Compute lambda matrix and "skew matrix" for frameified embedding
        if self.frameify:
            self.make_skew_mat()
        self.make_lambda()

    def make_skew_mat(self):
        """
        Compute skew matrix that converts embedding to a frame (when applied
        to vectors output by the raw embedding function)

        This is just an inverse square root of the Lambda, which satisfies
        S @ Lambda @ S.H = P, for P the projection onto the support of Lambda.
        """
        # Need to have the full lambda matrix
        self.make_lambda(shrink_mat=False)

        # Upgrade to double precision for the sensitive matrix operations
        original_dtype = self.raw_lamb_mat.dtype
        if self.raw_lamb_mat.is_complex():
            lamb_mat = self.raw_lamb_mat.to(torch.complex128)
        else:
            lamb_mat = self.raw_lamb_mat.double()

        # Skew mat can be taken as pinv of the (adjoint) Cholesky factor of lambda mat
        try:
            cholesky_factor = torch.linalg.cholesky(lamb_mat).T.conj()
            skew_mat = torch.linalg.pinv(cholesky_factor)
        except RuntimeError:
            eigvals, eigvecs = torch.linalg.eigh(lamb_mat)
            # Create pinv of square root of diagonal eigenvalue matrix
            rinv = 1 / eigvals.sqrt()
            rinv[eigvals < 1e-7] = 0
            skew_mat = eigvecs @ rinv.diag() @ eigvecs.T.conj()

        # Unskew lambda matrix, then save matrices as initial dtypes
        self.lamb_mat = (skew_mat.T.conj() @ lamb_mat @ skew_mat).to(original_dtype)
        self.skew_mat = skew_mat.to(original_dtype)

    def make_emb_mat(self, num_points: int = 1000):
        """
        Compute the lambda matrix used for normalization
        """
        # Compute the raw lambda matrix, computing number of points if needed
        if self.domain.continuous:
            points = torch.linspace(
                self.domain.min_val, self.domain.max_val, steps=num_points
            )
            self.num_points = num_points
            emb_vecs = self.emb_fun(points).to(self.dtype)
            assert emb_vecs.ndim == 2
            self.emb_dim = emb_vecs.shape[1]
            assert emb_vecs.shape[0] == num_points

            # Get rank-1 matrices for each point, then numerically integrate
            emb_mats = einsum("bi,bj->bij", emb_vecs, emb_vecs.conj())
            lamb_mat = torch.trapz(emb_mats, points, dim=0)

        else:
            points = torch.arange(self.domain.max_val).long()
            emb_vecs = self.emb_fun(points).to(self.dtype)
            assert emb_vecs.ndim == 2
            self.emb_dim = emb_vecs.shape[1]
            assert emb_vecs.shape[0] == self.domain.max_val

            # Get rank-1 matrices for each point, then sum together
            emb_mats = einsum("bi,bj->bij", emb_vecs, emb_vecs.conj())
            lamb_mat = torch.sum(emb_mats, dim=0)

        assert lamb_mat.ndim == 2
        assert lamb_mat.shape[0] == lamb_mat.shape[1]

        # For unframeified embeddings, use as regular lamb_mat
        lamb_mat = lamb_mat.to(self.dtype)
        self.raw_lamb_mat = lamb_mat
        if not self.frameify:
            self.lamb_mat = self.raw_lamb_mat

    def forward(self, input_data):
        """
        Embed input data via the user-specified embedding function
        """
        emb_vecs = self.emb_fun(input_data)

        # For frameified embeddings, skew matrix is applied to embedded vectors
        if hasattr(self, "frameify") and self.frameify:
            # Broadcast skew matrix so we can use batch matrix multiplication
            num_batch_dims = emb_vecs.ndim - 1
            emb_vecs = emb_vecs[..., None, :]
            skew_mat = self.skew_mat[(None,) * num_batch_dims]
            emb_vecs = torch.matmul(emb_vecs, skew_mat.conj())

            # Remove extra singleton dimension
            assert emb_vecs.size(-2) == 1
            return emb_vecs[..., 0, :]

        return emb_vecs


def make_emb_mats(emb_fun, start=0.0, stop=1.0, num_points=1000, eig_cutoff=1e-10):
    """
    Compute the three positive semidefinite matrices used during MPS training

    These three matrices are all powers of each other, and denoting them by
    E0, E1, and E2, we have E0 = E1.inv, E2 = E1 @ E1. E2 is in turn the matrix
    arising from integrating over all rank-1 embedding vectors
    """
    # Compute the raw embedding matrix E2
    points = np.linspace(start, stop, num=num_points)
    emb_vecs = emb_fun(points)
    assert emb_vecs.ndim == 2

    # Get rank-1 matrices for each point, then numerically integrate
    emb_mats = np.einsum("bi,bj->bij", emb_vecs, emb_vecs.conj())
    E2 = np.trapz(emb_mats, points, axis=0)
    assert E2.ndim == 2
    assert E2.shape[0] == E2.shape[1]

    # Compute eigendecomp of E2 and sqrt and (truncated) invsqrt of
    # eigenvalues, then use this to compute E0 and E1
    eigs, U = np.linalg.eigh(E2)
    cutoff_eigs = eigs < eig_cutoff
    sqrt_eigs = np.sqrt(eigs)
    invsqrt_eigs = 1 / sqrt_eigs
    invsqrt_eigs[cutoff_eigs] = 0
    E0 = U @ np.diag(invsqrt_eigs) @ U.T.conj()
    E1 = U @ np.diag(sqrt_eigs) @ U.T.conj()

    return E0, E1, E2


# def onehot_embed(tensor, emb_dim):
#     """
#     Function giving trivial one-hot embedding of categorical data
#     """
#     shape = tensor.shape + (emb_dim,)
#     output = np.zeros(shape)
#     output.scatter_(-1, tensor[..., None], 1)
#     return output


def binned_embed(tensor, emb_dim, start=0.0, stop=1.0):
    """
    Function binning continuous inputs into one-hot vectors
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy()
    orig_shape = tensor.shape
    orig_dtype = tensor.dtype
    numel = tensor.size

    # I can generalize to handle elements equal to `stop` if needed
    assert np.all(start <= tensor) and np.all(tensor <= stop)
    assert emb_dim <= 256

    # Rescale to lie in unit interval, and remove epsilon from elements at 1.
    tensor = (tensor - start) / (stop - start)
    tensor[tensor == 1.] = 0.9999

    # Bin into integral values
    tensor = np.floor(tensor * emb_dim).astype(np.uint8)

    # Do the actual one-hot indexing on flattened version of input
    tensor, idx = tensor.reshape(-1), np.arange(numel)
    out = np.zeros((numel, emb_dim), dtype=orig_dtype)
    out[idx, tensor] = 1

    return out.reshape(orig_shape + (emb_dim,))


# @torch.no_grad()
def trig_embed(data, emb_dim=2):
    r"""
    Function giving embedding from powers of sine and cosine

    Based on Equation B4 of E.M. Stoudenmire and D.J. Schwab, "Supervised
    Learning With Quantum-Inspired Tensor Networks", NIPS 2016, which maps an
    input x in the unit interval to a d-dim vector whose j'th component
    (where j = 0, 1, ..., d-1) is:

    .. math::
        \phi(x)_j = \sqrt{d-1 \choose j} \cos(\frac{pi}{2}x)^{d-j-1}
        \sin(\frac{pi}{2}x)^{j}

    Written by Raphaëlle Tihon
    """
    from scipy.special import binom

    emb_data = []
    for s in range(emb_dim):
        comp = (
            np.cos(data * np.pi / 2) ** (emb_dim - s - 1)
            * np.sin(data * np.pi / 2) ** s
        )
        comp *= np.sqrt(binom(emb_dim - 1, s))
        emb_data.append(comp)
    emb_data = np.stack(emb_data, axis=-1)
    assert emb_data.shape == data.shape + (emb_dim,)
    return emb_data


# @torch.no_grad()  # Implementation in Numpy
def legendre_embed(data, emb_dim=2):
    r"""
    Function giving embedding in terms of (orthonormal) Legendre polynomials

    Note that all polynomials are rescaled to lie on the unit interval, so that
    integrating the product of two polynomials over [0, 1] will give a
    Kronecker delta
    """
    # Function initializing Legendre polynomials over [0, 1] from coefficients
    leg_fun = partial(np.polynomial.Legendre, domain=[0.0, 1.0])

    emb_data = []
    original_dtype = data.dtype
    # data = data.numpy()
    base_coef = [0.0] * emb_dim
    for s in range(emb_dim):
        coef = base_coef.copy()
        coef[s] = 1.0
        raw_values = leg_fun(coef)(data)

        # Rescale the values to ensure each polynomial has unit norm (c.f.
        # https://en.wikipedia.org/wiki/Legendre_polynomials#Orthogonality_and_completeness)
        emb_data.append(sqrt(2 * s + 1) * raw_values)
    emb_data = np.stack(emb_data, axis=-1)
    assert emb_data.shape == data.shape + (emb_dim,)
    return emb_data.astype(original_dtype)


# def init_mlp_embed(
#     output_dim, num_layers=2, hidden_dims=[100], data_domain=None, frameify=False
# ):
#     """
#     Initialize multilayer perceptron embedding acting on scalar inputs

#     Args:
#         output_dim: Dimensionality of the embedded output vectors
#         num_layers: Total number of layers in the MLP embedding function
#         hidden_dims: List of dimensions of the hidden layers for the MLP
#         data_domain: If scalar inputs are not on the unit interval, a custom
#             data domain must be specified to allow correct normalization
#         frameify: Whether or not to rescale embedding vectors to ensure frame
#             condition is met throughout training
#     """
#     # Put all layer dimensions in single list
#     if isinstance(hidden_dims, int):
#         hidden_dims = [hidden_dims] * (num_layers - 1)
#     assert len(hidden_dims) == num_layers - 1
#     all_dims = [1] + list(hidden_dims) + [output_dim]

#     # Create underlying MLP from list of nn Modules
#     mod_list = []
#     for i in range(num_layers):
#         mod_list.append(nn.Linear(all_dims[i], all_dims[i + 1]))
#         mod_list.append(nn.Sigmoid())  # Sigmoid avoids putting outputs to 0
#     emb_fun = nn.Sequential(*mod_list)

#     # Initialize and return the trainable embedding function
#     if data_domain is None:
#         data_domain = unit_interval
#     return TrainableEmbedding(emb_fun, data_domain, frameify=frameify)
