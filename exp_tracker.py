#!/usr/bin/env python3
import os
import re
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path

### Helper functions invoked in the main routine


def setup_logging(log_file):
    import logging

    logging.basicConfig(
        filename=(log_file),
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    # Add logging to stdout (not just to log file)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    return logging


def setup(args):
    """
    Check user-specified arguments, assign exp id, and create exp directory
    """
    assert os.path.isfile(args.file)
    assert args.file.endswith(".py")  # Specialize to Python scripts for now
    assert args.mem_per_cpu > 0
    assert args.gpus >= 0

    # Make the top-level experiment directory if it doesn't exist yet
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    # Determine new experiment id, based on existing experiment directories
    exp_template = "experiment_{:03d}"
    pattern = re.compile(r"experiment_(\d+)")
    file_list = os.listdir(args.log_dir)
    # matches contains info about all previous valid experiment directories
    matches = list(filter(None, (pattern.match(f) for f in file_list)))
    if not matches:
        exp_id = 1
    else:
        exp_id = max(int(m.groups()[0]) for m in matches) + 1

    # Create new experiment directory and return reference to it
    exp_dir = Path(args.log_dir) / exp_template.format(exp_id)
    assert not os.path.exists(exp_dir)
    os.mkdir(exp_dir)
    return exp_dir


def copy_script(file, exp_dir):
    """
    Copy experimental script to experiment directory, with appended suffix
    """
    # Append the experiment id to the name of the source file
    base = Path(file).name[:-3]  # Remove .py suffix
    exp_id = str(exp_dir).split(sep="_")[-1]
    copy_path = exp_dir / f"{base}_{exp_id}.py"

    # Copy the file
    shutil.copy(file, copy_path)

    return copy_path


def to_slurm(args, exp_dir, run_script, script_args):
    """
    Send experimental script to Slurm (sbatch) with user-specified options

    If args.local is False, script will be run directly by Python interpreter
    """
    # Generate all the information for the call to sbatch
    LOG_DIR, LOG_FILE = str(exp_dir), str(exp_dir / "exp_record.log")
    if not args.local:
        slurm_call = [
            "sbatch",
            # Name of Slurm job
            f"--job-name=jorb_{str(exp_dir).split(sep='_')[-1]}",
            # Location of Slurm output file
            f"--output={str(exp_dir / 'slurm_log.out')}",
            # Number of GPUs
            f"--gpus={args.gpus}",
            # Minimum memory per CPU
            f"--mem-per-cpu={args.mem_per_cpu}G",
            # Run on unkillable partition by default
            f"--partition=unkillable",
            # Environment variables to pass to the experiment script
            f"--export=ALL,LOG_DIR={str(exp_dir)},LOG_FILE={str(exp_dir / 'exp_record.log')}",
        ]
        log_env = None
    else:
        slurm_call = ["python3"]
        # Pass modified environment to process
        log_env = os.environ.copy()
        log_env["LOG_DIR"], log_env["LOG_FILE"] = LOG_DIR, LOG_FILE

    # Path of script to be run, along with arguments to the script
    slurm_call.append(str(run_script))
    slurm_call += script_args

    # Call Slurm with the generated arguments
    return subprocess.run(slurm_call, env=log_env)


def main():
    # Grab all the user parameters from the command line
    parser = argparse.ArgumentParser(
        description="Run experiment scripts with Slurm, log the experimental setup"
    )
    parser.add_argument("file", type=str, help="Experiment file which will be run")
    parser.add_argument(
        "message", type=str, help="Message describing details of the experiment"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=str(Path(sys.path[0]) / "log_dir"),
        help="Location of the top-level experiment log directory",
    )
    parser.add_argument(
        "--mem-per-cpu",
        type=int,
        default=8,
        help="Minimum memory per CPU, in gigabytes",
    )
    parser.add_argument(
        "-G",
        "--gpus",
        type=int,
        default=0,
        help="Number of GPUs to allocate for experiment",
    )
    parser.add_argument(
        "--local",
        default=False,
        action="store_true",
        help="Whether to run the script locally, or via Slurm",
    )
    args, script_args = parser.parse_known_args()

    # Setup new directory for this experiment
    exp_dir = setup(args)

    # Save the general experiment info and Slurm arguments
    logging = setup_logging(exp_dir / "exp_record.log")
    logging.info("LOGGING CONFIGURATION")
    logging.info(json.dumps(vars(args), indent=4))
    logging.info(f"EXP_DIR = {str(exp_dir)}")

    # Save message in standalone file, for ease of reference
    with open(exp_dir / "message", "w") as f:
        f.write(args.message + "\n")

    # Copy the source file into the experiment directory
    copy_path = copy_script(args.file, exp_dir)

    # Send source file to Slurm (sbatch) and log call details
    try:
        call_info = to_slurm(args, exp_dir, copy_path, script_args)
        logging.info(call_info)
    except Exception as e:
        logging.error(f"to_slurm crashed, with message: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
