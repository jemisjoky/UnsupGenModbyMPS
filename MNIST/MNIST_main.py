# -*- coding: utf-8 -*-
import os
from sys import path, argv

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from MPScumulant import MPS_c, find_last_file, loadMPS

# path.append("../")
# mpl.use("Agg")
np.set_printoptions(5, linewidth=4 * 28)


def sample_image(mps, typ):
    dat = mps.generate_sample()
    a = int(np.sqrt(dat.size))
    img = dat.reshape((a, a))
    if typ == "s":
        for n in range(1, a, 2):
            img[n, :] = img[n, ::-1]
    return img


def sample_plot(mps, typ, nn):
    ncol = int(np.sqrt(nn))
    while nn % ncol != 0:
        ncol -= 1
    fig, axs = plt.subplots(nn // ncol, ncol)
    for ax in axs.flatten():
        ax.matshow(sample_image(mps, typ), cmap=mpl.cm.gray_r)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig("samples.pdf")


def loss_plot(mps, spars):
    fig, ax = plt.subplots()
    nsteps = 2 * mps.mps_len - 4
    if spars:
        ax.plot(np.arange(len(mps.losses)) * (nsteps // 2), mps.losses, ".")
    else:
        ax.plot(mps.losses)
    ax.xaxis.set_major_locator(MultipleLocator(nsteps))
    ax.xaxis.set_minor_locator(MultipleLocator(nsteps // 2))
    ax.xaxis.grid(which="both")
    ax.set_xticks([])
    plt.savefig("Loss.pdf")


def newest_exp():
    """
    Find the most recent MPS experiment in MPS experiments folder

    Returns the index of this MPS experiment
    """
    return find_last_file(run_dir, exp_prefix + r"(\d+)", return_path=False)


def find_latest_MPS(exp_folder):
    """
    Find the latest MPS checkpoint Pickle file in given MPS experiments folder

    Returns the index and path of the latest checkpoint file
    """
    return find_last_file(exp_folder, chk_prefix + r"(\d+)")


def init(warmup_loops=1):
    """Start the training, in a relatively high cutoff, over usually just 1 epoch"""
    dtset = np.load(dataset_name)

    # Create experiment directory
    if not os.path.isdir(run_dir):
        os.mkdir(run_dir)
    exp_num = newest_exp() + 1
    new_dir = run_dir + f"{exp_prefix}{exp_num}/"
    os.mkdir(new_dir)
    # with open("DATA_" + dataset_name.split("/")[-1] + ".txt", "w") as f:
    #  f.write(dataset_name)

    # Initialize model
    mps = MPS_c(28 * 28)
    mps.left_cano()
    mps.designate_data(dtset)
    mps.init_cumulants()
    mps.nbatch = 10
    mps.step_size = 0.05
    mps.descent_steps = 10
    mps.cutoff = 0.3

    # Train model for one loop, comparing initial and final losses
    loss = mps.get_train_loss()
    print(f"{init_header} {loss:.5f}")
    num_loops = 1
    cut_rec = mps.train(num_loops, rec_cut=True)
    mps.cutoff = cut_rec
    header_str = f"Loop {num_loops} loss:"
    header_str += " " * (len(init_header) - len(header_str))
    print(f"{header_str} {mps.losses[-1][1]:.5f}")

    save_dir = new_dir + f"{chk_prefix}{num_loops-1}/"
    os.mkdir(save_dir)
    mps.saveMPS(save_dir)

    return mps


def continue_train(lr_shrink, loopmax, safe_thres=0.5, lr_inf=1e-10):
    """
    Continue the training, in a fixed cutoff, train until loopmax is finished
    """
    exp_num = newest_exp()
    assert exp_num >= 0
    last_loop, folder = find_latest_MPS(run_dir + f"{exp_prefix}{exp_num}/")
    if last_loop > 0:
        print("Resuming:", folder)
    mps.loadMPS("Loop%dMPS" % last_loop)

    dtset = np.load(dataset_name)
    mps.designate_data(dtset)
    nlp = 5
    mps.verbose = 0
    # mps.descent_steps = 10
    mps.init_cumulants()
    # mps.verbose = 1
    # mps.cutoff = 1e-7

    """Set the hyperparameters here"""
    mps.maxibond = 800
    mps.nbatch = 20
    mps.descent_steps = 10
    mps.step_size = 0.001

    lr = mps.step_size
    while last_loop < loopmax:
        if mps.minibond > 1 and mps.bond_dimension.mean() > 10:
            mps.minibond = 1
            print("From now bondDmin=1")

        # train tentatively
        loss_last = mps.losses[-1][-1]
        while True:
            try:
                mps.train(nlp, rec_cut=False)
                if mps.losses[-1][-1] - loss_last > safe_thres:
                    print("lr=%1.3e is too large to continue safely" % lr)
                    raise Exception("lr=%1.3e is too large to continue safely" % lr)
            except:
                lr *= lr_shrink
                if lr < lr_inf:
                    print("lr becomes negligible.")
                    return
                mps.loadMPS("Loop%dMPS" % last_loop)
                mps.designate_data(dtset)
                mps.init_cumulants()
                mps.step_size = lr
            else:
                break

        last_loop += nlp
        assert False
        mps.saveMPS("Loop%d" % last_loop)  # TODO: Give this a real directory to save in
        print("Loop%d Saved" % last_loop)


if __name__ == "__main__":
    # Must be called with at least one command
    assert len(argv) > 1

    # Relevant directories for the random 1k images experiment
    mnist_dir = "./MNIST/"
    run_dir = mnist_dir + "rand1k_runs/"  # Location of experiment logs
    exp_prefix = "mnist1k_"  # Prefix for individual experiment directories
    chk_prefix = "Loop_"  # Prefix for individual experiment checkpoints
    dataset_name = mnist_dir + "mnist-rand1k_28_thr50_z/_data.npy"

    # Random global variables
    init_header = "Initial loss:"  # Header for printing initial loss

    if argv[1] == "init":
        mps = init()
        # locs = np.asarray([t[0] for t in mps.losses])
        # losses = np.asarray([t[1] for t in mps.losses])
        # plt.plot(locs, losses)
        # plt.show()
    elif argv[1] in ["train_from_scratch", "continue"]:
        # Initialize a new model if we're not continuing
        if argv[1] == "train_from_scratch":
            init(warmup_loops=0)
        if len(argv) > 2:
            num_loops = argv[2]
        else:
            num_loops = 250
        continue_train(0.9, num_loops, 0.05)
    # elif argv[1] == "plot":
    #     mps.loadMPS("./Loop%dMPS" % int(argv[2]))

    # # loss_plot(mps, True)
    # np.random.seed(1996)
    # sample_plot(mps, "z", 20)
