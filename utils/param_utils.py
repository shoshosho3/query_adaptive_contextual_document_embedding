import argparse

def get_args(for_query_adaptive_cde=True):

    """
    Get the arguments from the command line.
    :return: The parsed arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, help="BEIR dataset name.")
    parser.add_argument("--index", type=int, required=True, help="Index to check.")
    parser.add_argument("--seed", type=int, required=True, help="The seed of the program.")

    if for_query_adaptive_cde:
        parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden Dimension For the Model.")
        parser.add_argument("--epochs", type=int, required=True, help="Number of Epochs for Training Procedure.")

    return parser.parse_args()