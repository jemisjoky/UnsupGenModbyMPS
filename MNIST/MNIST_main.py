# -*- coding: utf-8 -*-
import os
import re
from sys import path, argv

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from MPScumulant import MPS_c, loadMPS

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


def find_last_file(search_dir, pattern, return_number=True, return_path=True):
    """
    Search among files or directories matching a pattern, find the last one

    The pattern will always involve some numerical substring, which is used as
    the ranking criteria. If no files exist with

    Args:
        search_dir: The directory which will be searched (non-recursively) for
            objects matching the desired pattern
        pattern: Regular expression containing a single group matching digits,
            to be fed into `re.compile`
        return_number: Whether to return the number associated with the file
        return_path: Whether to return the path to the file

    Returns:
        number: The number associated with the file
        path: The path of the file
    """
    if search_dir[-1] != "/":
        search_dir += "/"
    assert os.path.isdir(search_dir)
    assert return_number or return_path
    file_list = os.listdir(search_dir)
    regex = re.compile(pattern)
    matches = [regex.match(f) for f in file_list]
    if any(matches):
        idx, m = max(enumerate(matches), key=lambda x: int(x[1].groups()[0]))
        num = int(m.groups()[0])
        path = search_dir + file_list[idx]

        # Proper formatting for directories
        assert os.path.exists(path)
        if os.path.isdir(path) and path[-1] != "/":
            path += "/"
    else:
        num, path = -1, ""

    if return_number and return_path:
        return num, path
    elif return_number:
        return num
    else:
        return path


def find_latest_exp():
    """
    Find the most recent MPS experiment in MPS experiments folder

    Returns the index of this MPS experiment
    """
    return find_last_file(run_dir, exp_prefix + r"(\d+)")


def find_latest_MPS(exp_folder):
    """
    Find the latest MPS checkpoint Pickle file in given MPS experiments folder

    Returns the index and path of the latest checkpoint file
    """
    return find_last_file(exp_folder, chk_prefix + r"(\d+)" + chk_suffix)


def print_status(loop_num, mps):
    """
    Print the latest loss and bond dimensions attained by the MPS model
    """
    # MPS can be given as second arg, since loss is easy to extract
    assert isinstance(mps, MPS_c)

    # Dictionary containing all the info to print
    to_print = {}

    # Very first loss evaluation has different format
    if loop_num == 0:
        to_print["Initial loss:"] = mps.losses[-1][1]
    else:
        to_print[f"Loop {loop_num} loss:"] = mps.losses[-1][1]

    # Maximum and mean bond dimension
    bond_dims = mps.bond_dimension
    to_print["  Max bond dim:"] = max(bond_dims)
    to_print["  Mean bond dim:"] = sum(bond_dims) / len(bond_dims)

    # Do the actual printing
    head_width = max(len(k) for k in to_print.keys())
    float_format = "{0:<{w}} {1:.5f}"
    int_format = "{0:<{w}} {1}"
    for head, value in to_print.items():
        format_str = float_format if isinstance(value, float) else int_format
        print(format_str.format(head, value, w=head_width))


def init(warmup_loops=1):
    """Start the training, in a relatively high cutoff, over usually just 1 epoch"""
    dtset = np.load(dataset_name)

    # Create experiment directory
    if not os.path.isdir(run_dir):
        os.mkdir(run_dir)
    exp_num = find_latest_exp()[0] + 1
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
    mps.lr = 0.05
    mps.descent_steps = 10
    mps.cutoff = 0.3

    # Optionally train model, comparing initial and final losses
    mps.get_train_loss()
    print_status(0, mps)
    if warmup_loops > 0:
        cut_rec = mps.train(warmup_loops, rec_cut=True)
        mps.cutoff = cut_rec
        print_status(warmup_loops, mps)

    # mps.saveMPS(f"{new_dir}{chk_prefix}{warmup_loops}{chk_suffix}")
    mps.saveMPS(savefile_template.format(new_dir, warmup_loops))


def continue_train(lr_shrink, loopmax, safe_thres=0.5, lr_inf=1e-10):
    """
    Continue the training, in a fixed cutoff, train until loopmax is finished
    """
    # Find the current state of most recent MPS and load it
    exp_num, exp_folder = find_latest_exp()
    assert exp_num >= 0
    loop_num, save_path = find_latest_MPS(exp_folder)
    if loop_num > 0:
        print(f"Resuming: Loop {loop_num + 1}")
    mps = loadMPS(save_path, dataset_path=dataset_name)

    """Set the hyperparameters here"""
    nlp = 1
    mps.maxibond = 10
    mps.descent_steps = 10
    mps.lr = 0.001
    mps.cutoff = 1e-7
    # mps.nbatch = 20

    while loop_num < loopmax:
        # if mps.minibond > 1 and mps.bond_dimension.mean() > 10:
        #     mps.minibond = 1
        #     print("From now bondDmin=1")

        # Train the model while testing to see if learning rate is too large
        loss_last = mps.losses[-1][-1]
        good_lr = False
        while not good_lr:
            mps.train(nlp, rec_cut=False)
            if mps.losses[-1][-1] - loss_last > safe_thres:
                new_lr = mps.lr * lr_shrink
                print(f"lr={mps.lr:1.3e} is too large, decreasing to lr={new_lr:1.3e}")
                mps.lr = new_lr
                if mps.lr < lr_inf:
                    print("lr is negligible, ending training")
                    return

                # Load the last saved MPS and run it again with the reduced lr
                mps = loadMPS(find_latest_MPS(exp_folder)[1], dataset_path=dataset_name)
                mps.lr = lr
            else:
                good_lr = True
                loop_num += nlp
                mps.saveMPS(savefile_template.format(exp_folder, loop_num))
                print_status(loop_num, mps)
                if any(bd > 40 for bd in mps.bond_dimension):
                    mps.verbose = 2


if __name__ == "__main__":
    # Must be called with at least one command
    assert len(argv) > 1

    # Relevant directories for the random 1k images experiment
    mnist_dir = "./MNIST/"
    run_dir = mnist_dir + "rand1k_runs/"  # Location of experiment logs
    exp_prefix = "mnist1k_"  # Prefix for individual experiment directories
    chk_prefix = "mps_loop_"  # Prefix for individual experiment checkpoints
    chk_suffix = ".model.gz"  # Suffix for individual experiment checkpoints
    dataset_name = mnist_dir + "mnist-rand1k_28_thr50_z/_data.npy"
    savefile_template = "{}mps_loop_{:03d}.model.gz"

    if argv[1] == "init":
        mps = init()
    elif argv[1] in ["train_from_scratch", "continue"]:
        # Initialize a new model if we're not continuing
        if argv[1] == "train_from_scratch":
            init(warmup_loops=1)
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

    # locs = np.asarray([t[0] for t in mps.losses])
    # losses = np.asarray([t[1] for t in mps.losses])
    # plt.plot(locs, losses)
    # plt.show()
