# -*- coding: utf-8 -*-
import os
import re
import json
from time import time
from math import sqrt
from sys import argv

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from comet_ml import Experiment

from MPScumulant import MPS_c, loadMPS
from embeddings import trig_embed

# path.append("../")
# mpl.use("Agg")
np.set_printoptions(5, linewidth=4 * 28)


def sample_image(mps, typ):
    dat = mps.generate_sample()
    a = int(sqrt(dat.size))
    img = dat.reshape((a, a))
    if typ == "s":
        for n in range(1, a, 2):
            img[n, :] = img[n, ::-1]
    return img


def sample_plot(mps, typ, nn):
    ncol = int(sqrt(nn))
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
        idx, m = max(
            enumerate([m for m in matches if m]), key=lambda x: int(x[1].groups()[0])
        )
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
    return find_last_file(exp_folder, SAVEFILE_REGEX)


def print_status(loop_num, mps, test_loss, epoch_time):
    """
    Print the latest loss and bond dimensions attained by the MPS model
    """
    # MPS can be given as second arg, since loss is easy to extract
    assert isinstance(mps, MPS_c)

    # Dictionary containing all the info to print
    to_print = {}

    # Very first loss evaluation has different format
    if loop_num == 0:
        to_print["Initial train loss:"] = mps.losses[-1][1]
    else:
        to_print[f"Loop {loop_num} train loss:"] = mps.losses[-1][1]

    # Maximum and mean bond dimension
    to_print["  Test loss:"] = test_loss
    to_print["  Max bond dim:"] = max(mps.bond_dims)
    to_print["  Mean bond dim:"] = sum(mps.bond_dims) / len(mps.bond_dims)
    to_print["  Loop runtime:"] = epoch_time

    # Do the actual printing
    head_width = max(len(k) for k in to_print.keys())
    loss_format = "{0:<{w}} {1:.5f}"
    maxbd_format = "{0:<{w}} {1}"
    meanbd_format = "{0:<{w}} {1:.2f}"
    time_format = "{0:<{w}} {1:.1f}s"
    formats = [loss_format] * 2 + [maxbd_format, meanbd_format] + [time_format]
    for format_str, (head, value) in zip(formats, to_print.items()):
        print(format_str.format(head, value, w=head_width))


def init(warmup_loops=1):
    """Start the training, in a relatively high cutoff, over usually just 1 epoch"""
    # Create experiment directory
    if not os.path.isdir(RUN_DIR):
        os.mkdir(RUN_DIR)
    exp_num = find_latest_exp()[0] + 1
    exp_folder = RUN_DIR + f"{EXP_PREFIX}{exp_num}/"
    os.mkdir(exp_folder)

    # Initialize model
    mps = MPS_c(28 * 28)
    mps.designate_data(TRAIN_SET)
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

    # mps.saveMPS(f"{exp_folder}{chk_prefix}{warmup_loops}{chk_suffix}")
    if SAVE_MODEL:
        mps.saveMPS(SAVEFILE_TEMPLATE.format(exp_folder, warmup_loops))


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
        mps = loadMPS(save_path, dataset_path=TRAIN_SET_NAME)
        step_count = len(mps.losses)

    # Initialize a new MPS with desired hyperparameters
    else:
        # Create experiment directory
        if SAVE_MODEL:
            if not os.path.isdir(RUN_DIR):
                os.mkdir(RUN_DIR)
            exp_num = find_latest_exp()[0] + 1
            exp_folder = RUN_DIR + f"{EXP_PREFIX}{exp_num}/"
            os.mkdir(exp_folder)

            # Log the experimental parameters in the folder
            with open(f"{exp_folder}{EXP_NAME}.json", "w") as f:
                json.dump(PARAM_DICT, f, indent=4)

        # Initialize model
        start_time = time()
        mps = MPS_c(
            28 ** 2,
            cutoff=SV_CUTOFF,
            lr=LR,
            nbatch=NBATCH,
            verbose=VERBOSITY,
            max_bd=MAX_BDIM,
            min_bd=MIN_BDIM,
            init_bd=INIT_BDIM,
            seed=SEED,
            embed_fun=EMBEDDING_FUN,
        )
        mps.designate_data(TRAIN_SET.astype(np.float64))
        # mps.designate_data(TRAIN_SET)
        mps.get_train_loss()
        test_loss = mps.get_test_loss(TEST_SET)
        init_time = time() - start_time
        loop_num = 0
        step_count = 1  # Calcuation of initial training loss counts as step

        print_status(loop_num, mps, test_loss, init_time)
        if LOGGER is not None:
            LOGGER.log_metrics(
                {"train_loss": mps.losses[-1][-1], "test_loss": test_loss}, epoch=0
            )

    while loop_num < epochs:
        # if mps.minibond > 1 and mps.bond_dims.mean() > 10:
        #     mps.minibond = 1
        #     print("From now bondDmin=1")

        # Train the model while testing to see if learning rate is too large
        good_lr = False
        loss_last = mps.losses[-1][-1]
        while not good_lr:
            start_time = time()
            mps.train(num_epochs=1, rec_cut=False)
            epoch_time = time() - start_time

            # If loss is increasing, turn down LR and redo training
            if mps.losses[-1][-1] > loss_last and SAVE_MODEL:
                new_lr = mps.lr * LR_SHRINK
                print(f"lr={mps.lr:1.3e} is too large, decreasing to lr={new_lr:1.3e}")
                mps.lr = new_lr
                if mps.lr < MIN_LR:
                    print("lr is negligible, ending training")
                    return

                # Load the last saved MPS and run it again with the reduced lr
                mps = loadMPS(
                    find_latest_MPS(exp_folder)[1], dataset_path=TRAIN_SET_NAME
                )

            # Otherwise save the model and keep going
            else:
                good_lr = True
                loop_num += 1
                test_loss = mps.get_test_loss(TEST_SET)
                print_status(loop_num, mps, test_loss, epoch_time)
                if SAVE_MODEL:
                    last_loop, last_path = find_latest_MPS(exp_folder)
                    mps.saveMPS(SAVEFILE_TEMPLATE.format(exp_folder, loop_num))
                # if any(bd > 40 for bd in mps.bond_dims):
                #     mps.verbose = 2

                # If we're not keeping intermediate states, remove the last one
                if SAVE_MODEL and not SAVE_INTERMEDIATE:
                    this_loop, _ = find_latest_MPS(exp_folder)
                    if last_loop > -1:
                        assert this_loop == last_loop + 1
                        os.remove(last_path)

                # Log the data
                if LOGGER is not None:
                    assert len(mps.losses[step_count:]) % STEPS_PER_EPOCH == 0
                    for step, (bond, loss) in enumerate(mps.losses[step_count:]):
                        LOGGER.log_metrics(
                            {"current_bond": bond, "batch_loss": loss},
                            step=(step_count + step),
                            epoch=loop_num,
                        )
                    LOGGER.log_metrics(
                        {
                            "train_loss": mps.losses[-1][-1],
                            "test_loss": test_loss,
                            "max_bd": max(mps.bond_dims),
                            "min_bd": min(mps.bond_dims),
                            "mean_bd": sum(mps.bond_dims) / len(mps.bond_dims),
                        },
                        epoch=loop_num,
                    )

                step_count = len(mps.losses)


if __name__ == "__main__":
    # Must be called with at least one command
    assert len(argv) > 1

    ### Hyperparameters for the experiment ###
    # for MAX_BDIM in [50, 70, 100, 150, 200, 300, 400, 500, 750, 1000]:
    for MAX_BDIM in [10]:
        # MPS hyperparameters
        MIN_BDIM = 1
        # MAX_BDIM = 10
        INIT_BDIM = 2
        SV_CUTOFF = 1e-7
        EMBEDDING_FUN = trig_embed
        # EMBEDDING_FUN = None
        STEPS_PER_EPOCH = 2 * (28 ** 2) - 4

        # Training hyperparameters
        LR = 1e-3
        NBATCH = 10
        EPOCHS = 20
        VERBOSITY = 1
        LR_SHRINK = 0.95
        MIN_LR = 1e-10
        # COMET_LOG = True
        COMET_LOG = False
        # PROJECT_NAME = "hanetal-source-v2"
        PROJECT_NAME = "hanetal-source-v2"
        EXP_NAME = f"bd{MAX_BDIM}_bdi{INIT_BDIM}_cut{SV_CUTOFF:1.0e}"
        SAVE_MODEL = False
        SAVE_INTERMEDIATE = False
        ORIGINAL_DATASET = True
        SEED = 0

        # Save hyperparameters, setup Comet logger
        if "PARAM_DICT" in globals().keys():
            vars_to_delete = [
                "PARAM_DICT",
                "LOGGER",
                "MNIST_DIR",
                "RUN_DIR",
                "EXP_PREFIX",
                "TRAIN_SET_NAME",
                "TEST_SET_NAME",
                "SAVEFILE_TEMPLATE",
                "SAVEFILE_REGEX",
                "TRAIN_SET",
                "TEST_SET",
            ]
            assert all(var in globals().keys() for var in vars_to_delete)
            for var in vars_to_delete:
                del globals()[var]
        PARAM_DICT = {k.lower(): v for k, v in globals().items() if k.upper() == k}
        if COMET_LOG:
            LOGGER = Experiment(project_name=PROJECT_NAME)
            LOGGER.log_parameters(PARAM_DICT)
            LOGGER.set_name(EXP_NAME)
        else:
            LOGGER = None

        # Relevant directories for the random 1k images experiment
        MNIST_DIR = "./MNIST/"
        RUN_DIR = MNIST_DIR + "rand1k_runs/"  # Location of experiment logs
        EXP_PREFIX = "mnist1k_"  # Prefix for individual experiment directories
        TRAIN_SET_NAME = MNIST_DIR + "mnist-rand1k_28_thr50_z/paper_data.npy"
        # TRAIN_SET_NAME = MNIST_DIR + "mnist-rand1k_28_thr50_z/full_train.npy"
        TEST_SET_NAME = MNIST_DIR + "mnist-rand1k_28_thr50_z/first1k_test.npy"
        SAVEFILE_TEMPLATE = "{}mps_loop_{:03d}.model.gz"
        SAVEFILE_REGEX = r"mps_loop_(\d+)\.model\.gz"

        # Load 1000 random MNIST images
        TRAIN_SET = np.load(TRAIN_SET_NAME)
        TEST_SET = np.load(TEST_SET_NAME)

        if argv[1] == "init":
            mps = init()
        elif argv[1] in ["train_from_scratch", "continue"]:
            # Initialize a new model if we're not continuing
            continue_last = argv[1] == "continue"
            train(EPOCHS, continue_last=continue_last)

            # if argv[1] == "train_from_scratch":
            # init(warmup_loops=1)
        # elif argv[1] == "plot":
        #     mps.loadMPS("./Loop%dMPS" % int(argv[2]))

        # # loss_plot(mps, True)
        # np.random.seed(1996)
        # sample_plot(mps, "z", 20)

        # locs = np.asarray([t[0] for t in mps.losses])
        # losses = np.asarray([t[1] for t in mps.losses])
        # plt.plot(locs, losses)
        # plt.show()
