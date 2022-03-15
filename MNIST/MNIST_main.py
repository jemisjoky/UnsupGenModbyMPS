#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
from sys import argv
from time import time
from math import sqrt, log
from functools import partial

from comet_ml import Experiment
import torch
import numpy as np

# Make MPS functions available for import on any machine this might be run on
paths = ["/home/mila/m/millerja/UnsupGenModbyMPS", 
         "/home/jemis/Continuous-uMPS/UnsupGenModbyMPS"]
for p in paths[::-1]:
    sys.path.insert(0, p)

from MPScumulant import MPS_c, loadMPS
from MPScumulant_torch import MPS_c as MPS_c_torch
from MPScumulant_torch import loadMPS as loadMPS_torch
from exp_tracker import setup_logging
from embeddings import trig_embed, binned_embed

np.set_printoptions(5, linewidth=4 * 28)


def sample_image(mps, typ):
    dat = mps.generate_sample()
    a = int(sqrt(dat.size))
    img = dat.reshape((a, a))
    if typ == "s":
        for n in range(1, a, 2):
            img[n, :] = img[n, ::-1]
    return img


def print_status(loop_num, mps, test_loss, epoch_time, offset):
    """
    Print the latest loss and bond dimensions attained by the MPS model
    """
    # MPS can be given as second arg, since loss is easy to extract
    assert isinstance(mps, (MPS_c, MPS_c_torch))

    # Dictionary containing all the info to print
    to_print = {}

    # Very first loss evaluation has different format
    if loop_num == 0:
        to_print["Initial train loss:"] = mps.losses[-1][1] + offset
    else:
        to_print[f"Loop {loop_num} train loss:"] = mps.losses[-1][1] + offset

    # Maximum and mean bond dimension
    to_print["  Test loss:"] = test_loss + offset
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
        print_fun(format_str.format(head, value, w=head_width))


def train(
    epochs,
    continue_last=False,
):
    """
    Initialize and train MPS model on MNIST for given number of epochs
    """
    # Find the state of most recent MPS experiment and load it
    if continue_last:
        raise NotImplementedError
        # exp_num, exp_folder = find_latest_exp()
        # assert exp_num >= 0
        # loop_num, save_path = find_latest_MPS(exp_folder)
        # if loop_num > 0:
        #     print_fun(f"Resuming: Loop {loop_num + 1}")
        # mps = loadMPS(save_path, dataset_path=TRAIN_SET_NAME)
        # step_count = len(mps.losses)

    # Initialize a new MPS with desired hyperparameters
    else:
        # Create experiment directory
        if SAVE_MODEL:
            # Log the experimental parameters in the folder
            log_path = os.path.join(LOG_DIR, f"{EXP_NAME}.json")
            with open(log_path, "a") as f:
                json.dump(PARAM_DICT, f, indent=4)

        MPS = MPS_c_torch if USE_TORCH else MPS_c
        load_MPS = loadMPS_torch if USE_TORCH else loadMPS

        # Initialize model
        start_time = time()
        mps = MPS(
            28 ** 2,
            in_dim=IN_DIM,
            cutoff=SV_CUTOFF,
            lr=LR,
            nbatch=NBATCH,
            verbose=VERBOSITY,
            max_bd=MAX_BDIM,
            min_bd=MIN_BDIM,
            init_bd=INIT_BDIM,
            seed=SEED,
            embed_fun=EMBEDDING_FUN,
            device=DEVICE,
        )
        mps.designate_data(TRAIN_SET)
        mps.get_train_loss()
        train_loss = mps.get_test_loss(TRAIN_SET)
        test_loss = mps.get_test_loss(TEST_SET)
        init_time = time() - start_time
        loop_num = 0
        step_count = 1  # Calcuation of initial training loss counts as step

        # Add loss offset in case of embedded data
        offset = log(2) * (28 ** 2) if EMBEDDING_FUN is not None else 0.0

        print_fun(f"\n\nStarting config '{EXP_NAME}'")
        print_status(loop_num, mps, test_loss, init_time, offset)
        if LOGGER is not None:
            LOGGER.log_metrics(
                {"train_loss": train_loss + offset, "test_loss": test_loss + offset},
                epoch=0,
            )

    while loop_num < epochs:
        # if mps.minibond > 1 and mps.bond_dims.mean() > 10:
        #     mps.minibond = 1
        #     print_fun("From now bondDmin=1")

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
                print_fun(
                    f"lr={mps.lr:1.3e} is too large, decreasing to lr={new_lr:1.3e}"
                )
                if new_lr < MIN_LR:
                    print_fun("lr is negligible, ending training")
                    return

                # Load the last saved MPS and run it again with the reduced lr
                mps = load_MPS(mps_path, dataset_path=TRAIN_SET_NAME)
                mps.lr = new_lr

            # Otherwise save the model and keep going
            else:
                good_lr = True
                loop_num += 1
                test_loss = mps.get_test_loss(TEST_SET)
                mps_path = os.path.join(LOG_DIR, f"{EXP_NAME}_{loop_num}.model")
                if SAVE_MODEL:
                    mps.saveMPS(mps_path)
                    # Remove the last model state
                    if not SAVE_INTERMEDIATE and loop_num > 1:
                        old_path = os.path.join(
                            LOG_DIR, f"{EXP_NAME}_{loop_num-1}.model"
                        )
                        os.remove(old_path)

                # Add loss offset in case of embedded data
                print_status(loop_num, mps, test_loss, epoch_time, offset)

                # Log the data
                if LOGGER is not None:
                    assert len(mps.losses[step_count:]) % STEPS_PER_EPOCH == 0
                    for step, (bond, loss) in enumerate(mps.losses[step_count:]):
                        LOGGER.log_metrics(
                            {"current_bond": bond, "batch_loss": loss + offset},
                            step=(step_count + step),
                            epoch=loop_num,
                        )
                    LOGGER.log_metrics(
                        {
                            "train_loss": mps.losses[-1][-1] + offset,
                            "test_loss": test_loss + offset,
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

    #
    # Hyperparameters for the experiment
    #
    for MAX_BDIM in [10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 500, 750]:
        # MPS hyperparameters
        IN_DIM = 5
        MIN_BDIM = 2
        # MAX_BDIM = 10
        INIT_BDIM = 2
        SV_CUTOFF = 1e-7
        # EMBEDDING_FUN = None
        EMBEDDING_FUN = partial(trig_embed, emb_dim=IN_DIM)
        # EMBEDDING_FUN = partial(binned_embed, emb_dim=IN_DIM)
        STEPS_PER_EPOCH = 2 * (28 ** 2) - 4
        USE_TORCH = True

        # Training hyperparameters
        LR = 1e-3
        NBATCH = 10
        EPOCHS = 20
        VERBOSITY = 1
        LR_SHRINK = 9e-2
        MIN_LR = 1e-5
        COMET_LOG = True
        PROJECT_NAME = "hanetal-cluster-v2"
        SAVE_MODEL = True
        SAVE_INTERMEDIATE = False
        SEED = 0

        # Get info about the available GPUs
        n_gpu = torch.cuda.device_count()
        DEVICE = "cpu" if (n_gpu == 0 or not USE_TORCH) else "cuda:0"

        # Setup save directories and logging
        EXP_NAME = f"bd{MAX_BDIM}_id{IN_DIM}"
        EXP_NAME += ("_disc" if EMBEDDING_FUN is None else "_trig")
        EXP_NAME += ("" if DEVICE == "cpu" else "_gpu")
        LOG_DIR = os.getenv("LOG_DIR")
        LOG_FILE = os.getenv("LOG_FILE")
        assert (LOG_DIR is None) == (LOG_FILE is None)
        if LOG_DIR and "print_fun" not in globals():
            # Shadow print function with logging function
            logging = setup_logging(LOG_FILE)
            print_fun = logging.info
        else:
            print_fun = print
            # No model saving if we're running locally
            SAVE_MODEL = False

        # Save hyperparameters, setup Comet logger
        if "PARAM_DICT" in globals().keys():
            vars_to_delete = [
                "PARAM_DICT",
                "LOGGER",
                "MNIST_DIR",
                "EXP_PREFIX",
                "BIN_TRAIN_SET_NAME",
                "BIN_TEST_SET_NAME",
                "TRAIN_SET_NAME",
                "TEST_SET_NAME",
                "TRAIN_SET",
                "TEST_SET",
            ]
            assert all(var in globals().keys() for var in vars_to_delete)
            for var in vars_to_delete:
                del globals()[var]
        PARAM_DICT = {k.lower(): v for k, v in globals().items() if k.upper() == k}

        # Embedding function can't be serialized, so only keep whether nontrivial embed
        PARAM_DICT["embedding_fun"] = EMBEDDING_FUN is not None
        if COMET_LOG:
            LOGGER = Experiment(project_name=PROJECT_NAME)
            LOGGER.log_parameters(PARAM_DICT)
            LOGGER.set_name(EXP_NAME)
        else:
            LOGGER = None

        # Relevant directories for the random 1k images experiment
        MNIST_DIR = "./MNIST/"
        EXP_PREFIX = "mnist1k_"  # Prefix for individual experiment directories
        BIN_TRAIN_SET_NAME = MNIST_DIR + "mnist-rand1k_28_thr50_z/paper_data.npy"
        # BIN_TRAIN_SET_NAME = MNIST_DIR + "mnist-rand1k_28_thr50_z/full_train.npy"
        # BIN_TRAIN_SET_NAME = (
        #     MNIST_DIR + "mnist-rand1k_28_thr50_z/first1k_train_discrete.npy"
        # )
        BIN_TEST_SET_NAME = (
            MNIST_DIR + "mnist-rand1k_28_thr50_z/first1k_test_discrete.npy"
        )
        TRAIN_SET_NAME = MNIST_DIR + "mnist-rand1k_28_thr50_z/first1k_train.npy"
        TEST_SET_NAME = MNIST_DIR + "mnist-rand1k_28_thr50_z/first1k_test.npy"

        # Load 1000 random MNIST images
        TRAIN_SET = np.load(
            BIN_TRAIN_SET_NAME if EMBEDDING_FUN is None else TRAIN_SET_NAME
        )
        TEST_SET = np.load(
            BIN_TEST_SET_NAME if EMBEDDING_FUN is None else TEST_SET_NAME
        )
        if USE_TORCH:
            TRAIN_SET, TEST_SET = torch.tensor(TRAIN_SET), torch.tensor(TEST_SET)
            if EMBEDDING_FUN is None:
                TRAIN_SET, TEST_SET = TRAIN_SET.long(), TEST_SET.long()

        assert argv[1] in ["train_from_scratch", "continue"]
        # Initialize a new model if we're not continuing
        continue_last = argv[1] == "continue"
        train(EPOCHS, continue_last=continue_last)
