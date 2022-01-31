# -*- coding: utf-8 -*-
"""
class MPS cumulant
@author: congzlwag
"""
import os
import sys
import pickle

import gzip
import numpy as np
from numpy import einsum
from numpy.random import rand
from numpy.linalg import norm, svd


class MPS_c:
    def __init__(
        self,
        mps_len,
        in_dim=2,
        cutoff=0.01,
        lr=0.001,
        nbatch=10,
        verbose=1,
        max_bd=10,
        min_bd=1,
        init_bd=2,
        seed=1,
    ):
        """
        MPS class, with cumulant technique, efficient in DMRG-2
        Attributes:
            mps_len: length of chain
            in_dim: dimension of input to MPS cores
            cutoff: cutoff value setting minimum allowed singular value (as a
                fraction of the maximum singular value)
            lr: learning rate
            nbatch: number of batches to process the dataset in
            verbose: level of verbosity (0, 1, or 2)
            max_bd: maximum allowed bond dimension
            min_bd: minimum allowed bond dimension
            init_bd: dimension of bonds at initialization
            seed: random seed used to set model parameters

            bond_dims: list of bond dimensions, with bond_dims[i] connects i & i+1
            matrices: list of the tensors A^{(k)}
            merged_matrix:
                caches the merged order-4 tensor,
                when two adjacent tensors are merged but not decomposed yet
            current_bond: a multifunctional pointer
                -1: totally random matrices, need canonicalization
                in range(mps_len-1):
                    if merge_matrix is None: current_bond is the one to be merged next time
                    else: current_bond is the merged bond
                mps_len-1: left-canonicalized
            cumulant: see init_cumulants.__doc__

            losses: recorder of (current_bond, loss) tuples
            trainhistory: recorder of training history
        """
        np.random.seed(seed)
        self.mps_len = mps_len
        self.in_dim = in_dim
        self.cutoff = cutoff
        self.lr = lr
        self.nbatch = nbatch
        self.verbose = verbose
        assert min_bd <= init_bd
        self.min_bd = min_bd
        self.max_bd = max_bd

        # Initialize bond dimensions and MPS core tensors
        self.bond_dims = init_bd * np.ones((mps_len,), dtype=np.int16)
        self.bond_dims[-1] = 1
        self.matrices = [
            rand(self.bond_dims[i - 1], self.in_dim, self.bond_dims[i])
            for i in range(mps_len)
        ]

        self.current_bond = -1
        """Multifunctional pointer
        -1: totally random matrices, need canonicalization
        in range(mps_len-1): 
            if merge_matrix is None: current_bond is the one to be merged next time
            else: current_bond is the merged bond
        mps_len-1: left-canonicalized
        """
        self.merged_matrix = None
        self.losses = []
        self.trainhistory = []

        # Initialize matrices to be in left canonical form
        self.left_cano()

    def left_cano(self):
        """
        Canonicalizing all except the rightmost tensor left-canonical
        Can be called at any time
        """
        if self.merged_matrix is not None:
            self.rebuild_bond(True, keep_bdims=True)
        if self.current_bond == -1:
            self.current_bond = 0
        for bond in range(self.current_bond, self.mps_len - 1):
            self.merge_bond()
            self.rebuild_bond(going_right=True, keep_bdims=True)

    def merge_bond(self):
        k = self.current_bond
        self.merged_matrix = np.einsum(
            "ijk,klm->ijlm",
            self.matrices[k],
            self.matrices[(k + 1) % self.mps_len],
            order="C",
        )

    def rebuild_bond(self, going_right, spec=False, rec_cut=False, keep_bdims=False):
        """Decomposition
        going_right: if we're sweeping right or not
        keep_bdims: if the bond dimension is demanded to keep invariant compared with the value before being merged,
            when gauge transformation is carried out, this is often True.
        spec: if the truncated singular values are returned or not
        rec_cut: if a recommended cutoff is returned or not
        """
        assert self.merged_matrix is not None
        k = self.current_bond
        kp1 = (k + 1) % self.mps_len
        U, s, V = svd(
            self.merged_matrix.reshape(
                (
                    self.bond_dims[(k - 1) % self.mps_len] * self.in_dim,
                    self.in_dim * self.bond_dims[kp1],
                )
            )
        )

        if s[0] <= 0.0:
            print(
                "Error: At bond %d Merged_mat happens to be all-zero.\nPlease tune learning rate."
                % self.current_bond
            )
            raise FloatingPointError("Merged_mat trained to all-zero")

        if self.verbose > 1:
            print("bond:", k)
            # print(s)

        if not keep_bdims:
            bdmax = min(self.max_bd, s.size)
            new_bd = self.min_bd
            while new_bd < bdmax and s[new_bd] >= s[0] * self.cutoff:
                new_bd += 1
            # Found new_bd: s[:new_bd] dominate after cut off
        else:
            new_bd = self.bond_dims[k]
            # keep bond dimension

        if rec_cut:
            if new_bd >= bdmax:
                cut_recommend = -1.0
            else:
                cut_recommend = s[new_bd] / s[0]

        # Pad the singular values and singular matrices if needed
        if len(s) < new_bd:
            new_s = np.zeros((new_bd,), dtype=s.dtype)
            new_s[: len(s)] = s
            s = new_s
        else:
            s = s[:new_bd]
        if U.shape[1] < new_bd:
            new_U = np.zeros((U.shape[0], new_bd), dtype=U.dtype)
            new_U[:, : U.shape[1]] = U
            U = new_U
        else:
            U = U[:, :new_bd]
        if V.shape[0] < new_bd:
            new_V = np.zeros((new_bd, V.shape[1]), dtype=V.dtype)
            new_V[: V.shape[0]] = V
            V = new_V
        else:
            V = V[:new_bd, :]

        s = np.diag(s)
        bdm_last = self.bond_dims[k]
        self.bond_dims[k] = new_bd

        if going_right:
            V = np.dot(s, V)
            V /= norm(V)
        else:
            U = np.dot(U, s)
            U /= norm(U)

        if not keep_bdims:
            if self.verbose > 1:
                print("Bondim %d->%d" % (bdm_last, new_bd))

        self.matrices[k] = U.reshape(
            (self.bond_dims[(k - 1) % self.mps_len], self.in_dim, new_bd)
        )
        self.matrices[kp1] = V.reshape((new_bd, self.in_dim, self.bond_dims[kp1]))

        self.current_bond += 1 if going_right else -1
        self.merged_matrix = None

        if spec:
            if rec_cut:
                return np.diag(s), cut_recommend
            else:
                return np.diag(s)
        else:
            if rec_cut:
                return cut_recommend

    def designate_data(self, dataset):
        """Before the training starts, the training set is designated"""
        self.data = dataset.astype(np.int8)
        self.batchsize = self.data.shape[0] // self.nbatch
        self.init_cumulants()

    def init_cumulants(self):
        """
        Initialize a cache for left environments and right environments, `cumulants'
        During the training phase, it will be kept unchanged that:
        1) len(cumulant)== mps_len
        2) cumulant[0]  == np.ones((n_sample, 1))
        3) cumulant[-1] == np.ones((1, n_sample))
        4)  k = current_bond
            cumulant[j] =     if 0<j<=k: A(0)...A(j-1)
                            elif k<j<mps_len-1: A(j+1)...A(mps_len-1)
        """
        if self.current_bond == self.mps_len - 1:
            # In this case, the MPS is left-canonicalized except the right most one, so the bond to be merged is mps_len-2
            self.current_bond -= 1
        self.cumulants = [np.ones((self.data.shape[0], 1))]
        for n in range(0, self.current_bond):
            self.cumulants.append(
                einsum(
                    "ij,jik->ik",
                    self.cumulants[-1],
                    self.matrices[n][:, self.data[:, n], :],
                )
            )
        right_part = [np.ones((1, self.data.shape[0]))]
        for n in range(self.mps_len - 1, self.current_bond + 1, -1):
            right_part = [
                einsum(
                    "jil,li->ji", self.matrices[n][:, self.data[:, n]], right_part[0]
                )
            ] + right_part
        self.cumulants = self.cumulants + right_part

    def Give_psi_cumulant(self):
        """
        Calculate the probability amplitudes of everything in the training set
        """
        k = self.current_bond
        if self.merged_matrix is None:
            return einsum(
                "ij,jik,kil,li->i",
                self.cumulants[k],
                self.matrices[k][:, self.data[:, k], :],
                self.matrices[k + 1][:, self.data[:, k + 1], :],
                self.cumulants[k + 1],
            )
        else:
            return einsum(
                "ij,jik,ki->i",
                self.cumulants[k],
                self.merged_matrix[:, self.data[:, k], self.data[:, k + 1], :],
                self.cumulants[k + 1],
            )

    def get_train_loss(self, append=True):
        """Get the NLL averaged on the training set"""
        L = -np.log(np.abs(self.Give_psi_cumulant()) ** 2).mean()  # - self.data_shannon
        if append:
            self.losses.append([self.current_bond, L])
        return L

    # def calc_loss(self, dat):
    #     """Show the NLL averaged on an arbitrary set"""
    #     L = -np.log(np.abs(self.Give_psi(dat)) ** 2).mean()
    #     return L

    def gradient_descent_cumulants(self, batch_id):
        """ Gradient descent using cumulants, which efficiently avoids lots of tensor contraction!\\
            Together with update_cumulants, its computational complexity for updating each tensor is D^2
            Added by Pan Zhang on 2017.08.01
            Revised to single cumulant by Jun Wang on 20170802

        This trains the merged core containing the contraction of cores
        associated with sites k and k+1, for k := self.current_bond
        """
        # Index of the data in the current minibatch
        indx = range(batch_id * self.batchsize, (batch_id + 1) * self.batchsize)
        # All data in the current minibatch, with shape (batchsize, mps_len)
        states = self.data[indx]
        k = self.current_bond
        kp1 = (k + 1) % self.mps_len
        km1 = (k - 1) % self.mps_len
        #
        left_vecs = self.cumulants[k][indx, :]  # shape = (batchsize, D)
        right_vecs = self.cumulants[kp1][:, indx]  # shape = (D, batchsize)

        # Batch of rank-1 matrices containing left and right environments
        phi_mat = einsum("ij,ki->ijk", left_vecs, right_vecs)

        # Probability amplitudes associated with all inputs in the batch
        psi = einsum(
            "ij,jik,ki->i",
            left_vecs,
            self.merged_matrix[:, states[:, k], states[:, kp1], :],
            right_vecs,
        )

        gradient = np.zeros(
            (
                self.bond_dims[km1],
                self.in_dim,
                self.in_dim,
                self.bond_dims[kp1],
            )
        )
        psi_inv = 1 / psi
        if np.any(psi == 0):
            print(
                "Error: At bond %d, batchsize=%d, while %d of them psi=0."
                % (self.current_bond, self.batchsize, (psi == 0).sum())
            )
            print(np.argwhere(psi == 0).ravel())
            print("Maybe you should decrease n_batch")
            raise ZeroDivisionError("Some of the psis=0")

        # The gradient constructed below is equal to the sum of `batchsize`
        # individual contributions, each of which is an outer product of the
        # left and right environments for each datum, along with the (one-hot
        # encoded) input vectors associated with each datum at sites k and k+1.
        # Each one of these is rescaled by the inverse of the probability
        # amplitude associated with a datum, and then they are all averaged
        # together. A factor of the merged matrix is subtracted from
        # this before gradient before it is applied to the merged matrix, and
        # the result is finally normalized so the merged matrix has unit norm.
        for i, j in [(i, j) for i in range(self.in_dim) for j in range(self.in_dim)]:
            idx = (states[:, k] == i) * (states[:, kp1] == j)
            gradient[:, i, j, :] = (
                np.einsum("ijk,i->jk", phi_mat[idx, :, :], psi_inv[idx]) * 2
            )
        gradient /= self.batchsize
        gradient -= 2 * self.merged_matrix

        self.merged_matrix += gradient * self.lr
        self.merged_matrix /= norm(self.merged_matrix)

    def update_cumulants(self, gone_right_just_now):
        """After rebuid_bond, update self.cumulants.
        Bond has been rebuilt and self.current_bond has been changed,
        so it matters whether we have bubbled toward right or not just now
        """
        k = self.current_bond
        if gone_right_just_now:
            self.cumulants[k] = einsum(
                "ij,jik->ik",
                self.cumulants[k - 1],
                self.matrices[k - 1][:, self.data[:, k - 1], :],
            )
        else:
            self.cumulants[k + 1] = einsum(
                "jik,ki->ji",
                self.matrices[k + 2][:, self.data[:, k + 2], :],
                self.cumulants[k + 2],
            )

    def bond_train(self, going_right=True, rec_cut=False, calc_loss=True):
        """Training on current_bond
        going_right & rec_cut: see rebuild_bond
        calc_loss: whether get_train_loss is called.
        """
        self.merge_bond()
        batch_start = np.random.randint(self.nbatch)
        # batch_start = 0
        self.batchsize = self.data.shape[0] // self.nbatch
        for n in range(self.nbatch):
            self.gradient_descent_cumulants(batch_id=(batch_start + n) % self.nbatch)

        cut_recommend = self.rebuild_bond(going_right, rec_cut=rec_cut)
        self.update_cumulants(gone_right_just_now=going_right)
        if calc_loss:
            self.get_train_loss()
        if rec_cut:
            return cut_recommend

    def train(self, num_epochs=1, rec_cut=True):
        """
        Train over several epoches. `num_epochs' is the number of epoches
        """
        for loop in range(num_epochs):
            # cut_rec is recorded only during the last loop
            record_cr = rec_cut and (loop == num_epochs - 1)

            # Initial right-to-left optimization sweep
            for bond in range(self.mps_len - 2, 0, -1):
                out = self.bond_train(going_right=False, rec_cut=record_cr)
                if record_cr and bond == self.mps_len - 2:
                    cut_rec = [out]
                # self.log_batch_loss(self.losses[-1][-1], bond)
                # self.bond_train(going_right=False, calc_loss=(bond == 1))

            # Following left-to-right optimization sweep
            for bond in range(0, self.mps_len - 2):
                out = self.bond_train(going_right=True, rec_cut=record_cr)
                if record_cr:
                    cut_rec.append(out)
                # self.log_batch_loss(self.losses[-1][-1], bond)
                # going_right=True, calc_loss=(bond == self.mps_len - 3)

            # print("Current Loss: %.9f\nBondim:" % self.losses[-1][-1])
            # print(self.bond_dims)

        # All loops finished
        # print('Append History')
        self.trainhistory.append(
            (
                self.cutoff,
                num_epochs,
                self.lr,
                self.nbatch,
            )
        )
        if rec_cut:
            if self.verbose > 2:
                print(np.asarray(cut_rec))
            cut_rec.sort(reverse=True)
            k = max(5, int(self.mps_len * 0.2))
            while k >= 0 and cut_rec[k] < 0:
                k -= 1
            if k >= 0:
                # print("Recommend cutoff for next loop:", cut_rec[k])
                return cut_rec[k]
            else:
                # print("Recommend cutoff for next loop:", "Keep current value")
                return self.cutoff

    # def log_batch_loss(self, loss, bond, logger):
    #     """
    #     If logger is set, log the loss associated with local optimization step
    #     """
    #     if logger is not None:
    #         logger.log_metrics({"batch_loss": loss, "current_bond": bond})

    def train_snapshot(self):
        """
        Return a text summary of the state of training of the MPS
        """
        snapshot = []
        snapshot.append("Present State of MPS:")
        snapshot.append(
            "mps_len=%d,\ncutoff=%1.5e,\tlr=%1.5e,\tnbatch=%d"
            % (
                self.mps_len,
                self.cutoff,
                self.lr,
                self.nbatch,
            )
        )
        snapshot.append("bond dimensions:")
        a = int(np.sqrt(self.mps_len))
        if self.mps_len % a == 0:
            a *= len(str(self.bond_dims.max())) + 2
            snapshot.append(
                np.array2string(self.bond_dims, precision=0, max_line_width=a)
            )
        else:
            snapshot.append(np.array2string(self.bond_dims, precision=0))
        if len(self.losses) > 0:
            snapshot.append("loss=%1.6e" % self.losses[-1][-1])

        snapshot.append("cutoff\tn_loop\tn_descent\tlearning_rate\tn_batch")
        for cutoff, loops, lr, nbatch in self.trainhistory:
            snapshot.append(f"{cutoff:1.2e}\t{loops}\t{lr:1.2e}\t{nbatch}")

        return "\n".join(snapshot) + "\n\n"

    def saveMPS(self, save_path):
        """
        Saving compressed form of MPS object at designated path

        The dataset being trained on and cumulants derived from that dataset
        aren't included in the MPS saved on disk
        """
        assert self.merged_matrix is None

        # # Add to log file with high-level summary of state of training
        # with open(save_dir + "train.log", "a") as f:
        #     f.write(self.train_snapshot())

        # Store temporary attributes of MPS, so they can be restored post-save
        temp_attrs = ["cumulants", "data"]
        stash_dict = {attr: getattr(self, attr) for attr in temp_attrs}
        for attr in temp_attrs:
            delattr(self, attr)

        # Save the whole MPS in a pickle file
        # with open("mps.model", "wb") as f:
        with gzip.open(save_path, "wb") as f:
            pickle.dump(self, f)

        # Restore the temporary attributes
        for attr, value in stash_dict.items():
            setattr(self, attr, value)

    def Give_psi(self, states):
        """Calculate the corresponding psi for configuration `states'"""
        if states.ndim == 1:
            states = states.reshape((1, -1))
        if self.merged_matrix is not None:
            # There's a merged tensor
            nsam = states.shape[0]
            k = self.current_bond
            kp1 = (k + 1) % self.mps_len
            left_vecs = np.ones((nsam, 1))
            right_vecs = np.ones((1, nsam))
            for i in range(0, k):
                left_vecs = einsum(
                    "ij,jik->ik", left_vecs, self.matrices[i][:, states[:, i], :]
                )
            for i in range(self.mps_len - 1, k + 1, -1):
                right_vecs = einsum(
                    "jik,ki->ji", self.matrices[i][:, states[:, i], :], right_vecs
                )
            return einsum(
                "ik,kil,li->i",
                left_vecs,
                self.merged_matrix[:, states[:, k], states[:, kp1], :],
                right_vecs,
            )
        else:
            # TT -- default status
            # try:
            left_vecs = self.matrices[0][0, states[:, 0], :]
            # except IndexError:
            #     print(self.matrices[0].shape, states)
            #     sys.exit(-10)
            for n in range(1, self.mps_len - 1):
                left_vecs = einsum(
                    "ij,jil->il", left_vecs, self.matrices[n][:, states[:, n], :]
                )
            return einsum("ij,ji->i", left_vecs, self.matrices[-1][:, states[:, -1], 0])

    def Give_probab(self, states):
        """Calculate the corresponding probability for configuration `states'"""
        return np.abs(self.Give_psi(states)) ** 2

    def get_test_loss(self, test_set):
        """
        Get the NLL averaged on the test set
        """
        return -np.log(self.Give_probab(test_set)).mean()

    def generate_sample(self, given_seg=None, *arg):
        """
        Warning: This method has already been functionally covered by generate_sample_1, so it might be discarded in the future.
        Usage:
            1) Direct sampling: m.generate_sample()
            2) Conditioned sampling: m.generate_sample((l, r), array([s_l,s_{l+1},...,s_{r-1}]))
                array([s_l,s_{l+1},...,s_{r-1}]) is given, and (l,r) designates the location of this segment
        """
        state = np.empty((self.mps_len,), dtype=np.int8)
        if given_seg is None:
            if self.current_bond != self.mps_len - 1:
                print(
                    "Warning: MPS should have been left canonicalized, when generating samples"
                )
                self.left_cano()
                print("Left-canonicalized, but please add left_cano before generation.")
            vec = np.asarray([1])
            for p in range(self.mps_len - 1, -1, -1):
                vec_act = np.dot(self.matrices[p][:, 1], vec)
                if rand() < (norm(vec_act) / norm(vec)) ** 2:
                    state[p] = 1
                    vec = vec_act
                else:
                    state[p] = 0
                    vec = np.dot(self.matrices[p][:, 0], vec)
        else:
            l, r = given_seg
            # assign the given segment
            state[l:r] = arg[0][:]
            # canonicalization
            if self.current_bond > r - 1:
                for bond in range(self.current_bond, r - 2, -1):
                    self.merge_bond()
                    self.rebuild_bond(going_right=False, keep_bdims=True)
            elif self.current_bond < l:
                for bond in range(self.current_bond, l):
                    self.merge_bond()
                    self.rebuild_bond(going_right=True, keep_bdims=True)
            vec = self.matrices[l][:, state[l], :]
            for p in range(l + 1, r):
                vec = np.dot(vec, self.matrices[p][:, state[p], :])
                vec /= norm(vec)
            for p in range(r, self.mps_len):
                vec_act = np.dot(vec, self.matrices[p][:, 1])
                # if rand() < (norm(vec_act) / norm(vec))**2:
                if rand() < norm(vec_act) ** 2:
                    # activate
                    state[p] = 1
                    vec = vec_act
                else:
                    # keep 0
                    state[p] = 0
                    vec = np.dot(vec, self.matrices[p][:, 0])
                vec /= norm(vec)
            for p in range(l - 1, -1, -1):
                vec_act = np.dot(self.matrices[p][:, 1], vec)
                # if rand() < (norm(vec_act) / norm(vec))**2:
                if rand() < norm(vec_act) ** 2:
                    state[p] = 1
                    vec = vec_act
                else:
                    state[p] = 0
                    vec = np.dot(self.matrices[p][:, 0], vec)
                vec /= norm(vec)
        return state

    def generate_sample_1(self, stat=None, givn_msk=None):
        """
        This direct sampler generate one sample each time.
        We highly recommend to canonicalize the MPS such that the only uncanonical bit is given,
        because when conducting mass sampling, canonicalization will be an unnecessary overhead!
        Usage:
            If the generation starts from scratch, just keep stat=None and givn_msk=None;
            else please assign
                givn_msk: an numpy.array whose shape is (mps_len,) and dtype=bool
                to specify which of the bits are given, and
                stat: an numpy.array whose shape is (mps_len,) and dtype=numpy.int8
                to specify the configuration of the given bits, the other bits will be ignored.

        """
        # <<<case: Start from scratch
        if stat is None or givn_msk is None or givn_msk.any() == False:
            if self.current_bond != self.mps_len - 1:
                self.left_cano()
            state = np.empty((self.mps_len,), dtype=np.int8)
            vec = np.asarray([1])
            for p in np.arange(self.mps_len)[::-1]:
                vec_act = np.dot(self.matrices[p][:, 1], vec)
                if rand() < (norm(vec_act) / norm(vec)) ** 2:
                    state[p] = 1
                    vec = vec_act
                else:
                    state[p] = 0
                    vec = np.dot(self.matrices[p][:, 0], vec)
            return state
        # case: Start from scratch>>>

        state = stat.copy()
        state[givn_msk == False] = -1
        givn_mask = givn_msk.copy()
        p = self.mps_len - 1
        while givn_mask[p] == False:
            p -= 1
        p_uncan = p
        # p_uncan points on the rightmost given bit

        """Canonnicalizing the MPS into mix-canonical form that the only uncanonical tensor is at p_uncan
        There's a bit trouble, for the uncanonical tensor is not recorded.
        It can be on either the left or the right of current_bond, 
        so we firstly check whether the bits on both sides of current_bond are given or not
        """
        bd = self.current_bond
        if givn_mask[bd] == False or givn_mask[bd + 1] == False:
            # not both of the bits connected by current_bond are given, so we need to canonicalize the MPS
            if bd >= p_uncan:
                while self.current_bond >= p_uncan:
                    self.merge_bond()
                    self.rebuild_bond(False, keep_bdims=True)
            else:
                while self.current_bond < p_uncan:
                    self.merge_bond()
                    self.rebuild_bond(False, keep_bdims=True)
        """Canonicalization finished
        From now on we should never operate the matrices in the sampling
        """
        plft = 0
        while givn_mask[plft] == False:
            plft += 1
        # plft points on the leftmost given bit

        p = plft
        while p < self.mps_len and givn_mask[p]:
            p += 1
        plft2 = p - 1
        # Since plft is a given bit, there's a segment of bits that plft is in. plft2 points on the right edge of this segment

        # <<< If there's no intermediate bit that need to be sampled
        if plft2 == p_uncan:
            vec = self.matrices[plft][:, state[plft], :]
            for p in range(plft + 1, plft2 + 1):
                vec = np.dot(vec, self.matrices[p][:, state[p], :])
                vec /= norm(vec)
            for p in range(plft2 + 1, self.mps_len):
                vec_act = np.dot(vec, self.matrices[p][:, 1])
                nom = norm(vec_act)
                if rand() < nom ** 2:
                    # activate
                    state[p] = 1
                    vec = vec_act / nom
                else:
                    # keep 0
                    state[p] = 0
                    vec = np.dot(vec, self.matrices[p][:, 0])
                    vec /= norm(vec)
            for p in np.arange(plft)[::-1]:
                vec_act = np.dot(self.matrices[p][:, 1], vec)
                nom = norm(vec_act)
                if rand() < nom ** 2:
                    state[p] = 1
                    vec = vec_act / nom
                else:
                    state[p] = 0
                    vec = np.dot(self.matrices[p][:, 0], vec)
                    vec /= norm(vec)
            # assert (state!=-1).all()
            return state
        # >>>

        """Dealing with the intermediated ungiven bits, sampling from plft2 to p_uncan. Only ungiven bits are sampled, of course.
        Firstly, we need to prepare
            left_vec: a growing ladder-shape TN, accumulatedly multiplied from plft to the right edge of the given segment plft is in.
            right_vecs: list of ladder-shape TNs
                right_vecs[p] is the TN accumulately multiplied from p_uncan to p (including p)
        """
        left_vec = einsum(
            "kj,kl->jl",
            self.matrices[plft][:, state[plft]],
            self.matrices[plft][:, state[plft]],
        )
        left_vec /= np.trace(left_vec)
        for p in range(plft + 1, plft2 + 1):
            left_vec = einsum(
                "pl,lq->pq",
                einsum("jl,jp->pl", left_vec, self.matrices[p][:, state[p]]),
                self.matrices[p][:, state[p]],
            )
            left_vec /= np.trace(left_vec)

        right_vecs = np.empty((self.mps_len), dtype=object)
        p = p_uncan
        right_vecs[p] = einsum(
            "ij,kj->ik", self.matrices[p][:, state[p]], self.matrices[p][:, state[p]]
        )
        right_vecs[p] /= np.trace(right_vecs[p])
        p -= 1
        while p > plft2:
            if givn_mask[p]:
                right_vecs[p] = einsum(
                    "qk,pk->pq",
                    self.matrices[p][:, state[p]],
                    einsum(
                        "pi,ik->pk", self.matrices[p][:, state[p]], right_vecs[p + 1]
                    ),
                )
            else:
                right_vecs[p] = einsum(
                    "qjk,pjk->pq",
                    self.matrices[p],
                    einsum("pji,ik->pjk", self.matrices[p], right_vecs[p + 1]),
                )
            right_vecs[p] /= np.trace(right_vecs[p])
            p -= 1

        # Secondly, sample the intermediate bits
        p = plft2 + 1
        while p <= p_uncan:
            if not givn_mask[p]:
                prob_marg = einsum(
                    "pq,pq",
                    einsum(
                        "pil,liq->pq",
                        einsum("jl,jip->pil", left_vec, self.matrices[p]),
                        self.matrices[p],
                    ),
                    right_vecs[p + 1],
                )
                left_vec_act = einsum(
                    "pl,lq->pq",
                    einsum("jl,jp->pl", left_vec, self.matrices[p][:, 1]),
                    self.matrices[p][:, 1],
                )
                prob_actv = einsum("pq,pq", left_vec_act, right_vecs[p + 1])
                if rand() < prob_actv / prob_marg:
                    state[p] = 1
                    left_vec = left_vec_act
                else:
                    state[p] = 0
                    left_vec = einsum(
                        "pl,lq->pq",
                        einsum("jl,jp->pl", left_vec, self.matrices[p][:, 0]),
                        self.matrices[p][:, 0],
                    )
                givn_mask[p] = True
            else:
                left_vec = einsum(
                    "pl,lq->pq",
                    einsum("jl,jp->pl", left_vec, self.matrices[p][:, state[p]]),
                    self.matrices[p][:, state[p]],
                )
            left_vec /= np.trace(left_vec)
            p += 1

        # Sampling the ungiven segment that connects to the right end
        while p < self.mps_len:
            left_vec_act = einsum(
                "pl,lq->pq",
                einsum("jl,jp->pl", left_vec, self.matrices[p][:, 1]),
                self.matrices[p][:, 1],
            )
            prob_actv = np.trace(left_vec_act)
            if rand() < prob_actv:
                state[p] = 1
                left_vec = left_vec_act
                left_vec /= prob_actv
            else:
                state[p] = 0
                left_vec = einsum(
                    "pl,lq->pq",
                    einsum("jl,jp->pl", left_vec, self.matrices[p][:, 0]),
                    self.matrices[p][:, 0],
                )
                left_vec /= np.trace(left_vec)
            givn_mask[p] = True
            p += 1
        # Sample the ungiven segment that connects to the left end
        if plft == 0:
            # bit 0 is given
            return state
        # else recursively generate
        return self.generate_sample_1(state, givn_mask)

    def proper_cano(self, target_bond, update_cumulant):
        """Gauge Transform the MPS into the mix-canonical form that:
        the only uncanonical tensor is either target_bond or target_bond+1. Both are accepted.
        """
        if self.current_bond == target_bond:
            return
        else:
            direction = 1 if self.current_bond < target_bond else -1
            for b in range(self.current_bond, target_bond, direction):
                self.merge_bond(b)
                self.rebuild_bond(going_right=(direction == 1), keep_bdims=True)
                if update_cumulant:
                    self.update_cumulants(direction == 1)


def loadMPS(save_file, dataset_path=None):
    """
    Load a saved MPS from a Pickle file and possibly initialize it with data
    """
    with gzip.open(save_file, "rb") as f:
        mps = pickle.load(f)

    if dataset_path is not None:
        dataset = np.load(dataset_path)
        mps.designate_data(dataset)

    return mps
