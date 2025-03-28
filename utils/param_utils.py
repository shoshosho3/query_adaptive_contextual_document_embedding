import argparse

def get_args(for_query_adaptive_cde=True):

    """
    Get the arguments from the command line.
    :return: The parsed arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, help="BEIR dataset name.")
    parser.add_argument("--index", type=int, required=True, help="Index to check.")

    if for_query_adaptive_cde:
        parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden Dimension For the Model.")
        parser.add_argument("--epochs", type=int, required=True, help="Number of Epochs for Training Procedure.")
        parser.add_argument("--seed", type=int, required=True, help="The seed of the program.")

    return parser.parse_args()


def set_seed(seed):

    """
    Set the seed for the program.
    :param seed: The seed to set.
    """

    import torch
    import random
    import numpy as np

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


def exists(dataset, seed):

    """
    Check if the dataset already exists.
    :param dataset: The dataset to check.
    :param seed: The seed to check.
    :return: True if the dataset exists, False otherwise.
    """

    import os

    return os.path.exists(f"{dataset}/doc_embeddings_{dataset}_{seed}")
