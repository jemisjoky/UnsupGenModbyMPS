# -*- coding: utf-8 -*-
import os
import re
from sys import path, argv

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from comet_ml import Experiment

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
    return find_last_file(RUN_DIR, EXP_PREFIX + r"(\d+)")


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
    to_print["  Max bond dim:"] = max(mps.bond_dims)
    to_print["  Mean bond dim:"] = sum(mps.bond_dims) / len(mps.bond_dims)

    # Do the actual printing
    head_width = max(len(k) for k in to_print.keys())
    float_format = "{0:<{w}} {1:.5f}"
    int_format = "{0:<{w}} {1}"
    for head, value in to_print.items():
        format_str = float_format if isinstance(value, float) else int_format
        print(format_str.format(head, value, w=head_width))


def init(warmup_loops=1):
    """Start the training, in a relatively high cutoff, over usually just 1 epoch"""
    # Create experiment directory
    if not os.path.isdir(RUN_DIR):
        os.mkdir(RUN_DIR)
    exp_num = find_latest_exp()[0] + 1
    new_dir = RUN_DIR + f"{EXP_PREFIX}{exp_num}/"
    os.mkdir(new_dir)

    # Initialize model
    mps = MPS_c(28 * 28)
    mps.designate_data(DATASET)
    mps.nbatch = 10
    mps.lr = 0.05
    mps.cutoff = 0.3

    # Optionally train model, comparing initial and final losses
    mps.get_train_loss()
    print_status(0, mps)
    if warmup_loops > 0:
        cut_rec = mps.train(num_epochs=warmup_loops, rec_cut=True)
        mps.cutoff = cut_rec
        print_status(warmup_loops, mps)

    # mps.saveMPS(f"{new_dir}{chk_prefix}{warmup_loops}{chk_suffix}")
    mps.saveMPS(SAVEFILE_TEMPLATE.format(new_dir, warmup_loops))


def train(
    epochs,
    continue_last=False,
):
    """
    Initialize and train MPS model on MNIST for given number of epochs
    """
    # Find the state of most recent MPS experiment and load it
    if continue_last:
        exp_num, exp_folder = find_latest_exp()
        assert exp_num >= 0
        loop_num, save_path = find_latest_MPS(exp_folder)
        if loop_num > 0:
            print(f"Resuming: Loop {loop_num + 1}")
        mps = loadMPS(save_path, dataset_path=DATASET_NAME)

    # Initialize a new MPS with desired hyperparameters
    else:
        # Create experiment directory
        if not os.path.isdir(RUN_DIR):
            os.mkdir(RUN_DIR)
        exp_num = find_latest_exp()[0] + 1
        new_dir = RUN_DIR + f"{EXP_PREFIX}{exp_num}/"
        os.mkdir(new_dir)

        # Initialize model
        mps = MPS_c(
            28 ** 2,
            cutoff=SV_CUTOFF,
            lr=LR,
            verbose=VERBOSITY,
            max_bd=MAX_BDIM,
            min_bd=MIN_BDIM,
            init_bd=INIT_BDIM,
            seed=SEED,
            logger=LOGGER,
        )
        mps.designate_data(DATASET)

    """Set the hyperparameters here"""
    nlp = 1
    mps.maxibond = 10
    mps.lr = 0.001
    mps.cutoff = 1e-7
    mps.nbatch = 10

    while loop_num < epochs:
        # if mps.minibond > 1 and mps.bond_dims.mean() > 10:
        #     mps.minibond = 1
        #     print("From now bondDmin=1")

        # Train the model while testing to see if learning rate is too large
        loss_last = mps.losses[-1][-1]
        good_lr = False
        while not good_lr:
            mps.train(num_epochs=1, rec_cut=False)

            # If loss is increasing, turn down LR and redo training
            if mps.losses[-1][-1] > loss_last:
                new_lr = mps.lr * LR_SHRINK
                print(f"lr={mps.lr:1.3e} is too large, decreasing to lr={new_lr:1.3e}")
                mps.lr = new_lr
                if mps.lr < MIN_LR:
                    print("lr is negligible, ending training")
                    return

                # Load the last saved MPS and run it again with the reduced lr
                mps = loadMPS(find_latest_MPS(exp_folder)[1], dataset_path=DATASET_NAME)
                mps.lr = lr

            # Otherwise save the model and keep going
            else:
                good_lr = True
                loop_num += nlp
                mps.saveMPS(SAVEFILE_TEMPLATE.format(exp_folder, loop_num))
                print_status(loop_num, mps)
                # if any(bd > 40 for bd in mps.bond_dims):
                #     mps.verbose = 2


if __name__ == "__main__":
    # Must be called with at least one command
    assert len(argv) > 1

    ### Hyperparameters for the experiment ###
    # MPS hyperparameters
    MIN_BDIM = 1
    MAX_BDIM = 10
    INIT_BDIM = 2
    SV_CUTOFF = 1e-7

    # Training hyperparameters
    LR = 1e-3
    EPOCHS = 100
    VERBOSITY = 1
    LR_SHRINK = 0.95
    MIN_LR = 1e-10
    COMET_LOG = False
    EXP_NAME = "hanetal-source-v1"
    SAVE_INTERMEDIATE = False

    # Save hyperparameters, setup Comet logger
    param_dict = {k.lower(): v for k, v in globals().items() if k.upper() == k}
    if COMET_LOG:
        LOGGER = Experiment(project_name=EXP_NAME)
        LOGGER.log_parameters(param_dict)
    else:
        LOGGER = None

    # Relevant directories for the random 1k images experiment
    MNIST_DIR = "./MNIST/"
    RUN_DIR = MNIST_DIR + "rand1k_runs/"  # Location of experiment logs
    EXP_PREFIX = "mnist1k_"  # Prefix for individual experiment directories
    DATASET_NAME = MNIST_DIR + "mnist-rand1k_28_thr50_z/_data.npy"
    SAVEFILE_TEMPLATE = "{}mps_loop_{:03d}.model.gz"

    # Load 1000 random MNIST images
    DATASET = np.load(DATASET_NAME)

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
