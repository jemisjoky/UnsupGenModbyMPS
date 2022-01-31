import os


def rm_intermediate_checkpoints(exp_dir="./MNIST/rand1k_runs/"):
    """
    Delete all experiment checkpoints except the final one
    """
    for folder in os.listdir(exp_dir):
        folder = f"{exp_dir}{folder}/"
        file_list = sorted(os.listdir(folder))
        for file in file_list[:-1]:
            if file.endswith(".json"):
                continue
            os.remove(folder + file)


if __name__ == "__main__":
    from sys import argv

    assert len(argv) > 1

    if argv[1] == "rm_intermediate":
        rm_intermediate_checkpoints()
