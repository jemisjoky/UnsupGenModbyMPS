# -*- coding: utf-8 -*-
from sys import path, argv

# path.append("../")

from MPScumulant import MPS_c
import os
import re
import numpy as np

np.set_printoptions(5, linewidth=4 * 28)
from time import strftime
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


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
    nsteps = 2 * mps.space_size - 4
    if spars:
        ax.plot(np.arange(len(mps.losses)) * (nsteps // 2), mps.losses, ".")
    else:
        ax.plot(mps.losses)
    ax.xaxis.set_major_locator(MultipleLocator(nsteps))
    ax.xaxis.set_minor_locator(MultipleLocator(nsteps // 2))
    ax.xaxis.grid(which="both")
    ax.set_xticks([])
    plt.savefig("Loss.pdf")


def find_latest_MPS():
    """Searching for the last MPS in current directory"""
    breakpoint()
    name_list = os.listdir(run_dir)
    pmx = -1
    mx = 0
    pattrn = r"Loop(\d+)M"
    for prun in range(len(name_list)):
        mach = re.match(pattrn, name_list[prun])
        if mach is not None:
            nl = int(mach.group(1))
            if nl >= mx:
                pmx = prun
                mx = nl
    if pmx == -1:
        print("No MPS Found")
    else:
        return mx, name_list[pmx]


def start():
    """Start the training, in a relatively high cutoff, over usually just 1 epoch"""
    dtset = np.load(dataset_name)

    new_dir = run_dir + strftime("mnist_%Y_%m_%d_%H%M") + "/"
    os.mkdir(new_dir)
    # with open("DATA_" + dataset_name.split("/")[-1] + ".txt", "w") as f:
    #  f.write(dataset_name)

    # Initialize model
    m.left_cano()
    m.designate_data(dtset)
    m.init_cumulants()
    m.nbatch = 10
    m.descenting_step_length = 0.05
    m.descent_steps = 10
    m.cutoff = 0.3

    loss = m.get_train_loss()
    print(f"Initial loss: {loss:.5f}")
    num_loops = 1
    cut_rec = m.train(num_loops, rec_cut=True)
    m.cutoff = cut_rec
    save_dir = new_dir + f"Loop{num_loops-1}/"
    os.mkdir(save_dir)
    m.saveMPS(save_dir)


def onecutrain(lr_shrink, loopmax, safe_thres=0.5, lr_inf=1e-10):
    """Continue the training, in a fixed cutoff, train until loopmax is finished"""
    assert False  # dtset = np.load("../" + dataset_name)
    m.designate_data(dtset)

    mx, folder = find_latest_MPS()
    print("Resuming: ", folder)

    loop_last = mx
    nlp = 5
    m.verbose = 0
    m.loadMPS("Loop%dMPS" % loop_last)
    # m.descent_steps = 10
    m.init_cumulants()
    # m.verbose = 1
    # m.cutoff = 1e-7

    """Set the hyperparameters here"""
    m.maxibond = 800
    m.nbatch = 20
    m.descent_steps = 10
    m.descenting_step_length = 0.001

    lr = m.descenting_step_length
    while loop_last < loopmax:
        if m.minibond > 1 and m.bond_dimension.mean() > 10:
            m.minibond = 1
            print("From now bondDmin=1")

        # train tentatively
        loss_last = m.losses[-1]
        while True:
            try:
                m.train(nlp, rec_cut=False)
                if m.losses[-1] - loss_last > safe_thres:
                    print("lr=%1.3e is too large to continue safely" % lr)
                    raise Exception("lr=%1.3e is too large to continue safely" % lr)
            except:
                lr *= lr_shrink
                if lr < lr_inf:
                    print("lr becomes negligible.")
                    return
                m.loadMPS("Loop%dMPS" % loop_last)
                m.designate_data(dtset)
                m.init_cumulants()
                m.descenting_step_length = lr
            else:
                break

        loop_last += nlp
        assert False
        m.saveMPS("Loop%d" % loop_last)  # TODO: Give this a real directory to save in
        print("Loop%d Saved" % loop_last)


if __name__ == "__main__":
    import traceback
    import warnings
    import sys

    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

        log = file if hasattr(file, "write") else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = warn_with_traceback

    # Must be called with at least one command
    assert len(argv) > 1

    # Relevant directories for the random 1k images experiment
    mnist_dir = "./MNIST/"
    run_dir = mnist_dir + "rand1k_runs/"
    dataset_name = mnist_dir + "mnist-rand1k_28_thr50_z/_data.npy"

    m = MPS_c(28 * 28)

    if argv[1] == "init":
        start()
    elif argv[1] == "one":
        onecutrain(0.9, 250, 0.05)
    elif argv[1] == "plot":
        m.loadMPS("./Loop%dMPS" % int(argv[2]))
    elif argv[1] == "latest_name":
        print(find_latest_MPS())

    # # loss_plot(m, True)
    # np.random.seed(1996)
    # sample_plot(m, "z", 20)
