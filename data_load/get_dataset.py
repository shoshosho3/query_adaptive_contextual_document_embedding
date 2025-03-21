import os
from beir import util

# URL to download datasets from the BEIR repository
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"

# Directory where datasets are stored
datasets_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")


def does_exist(dataset_name: str):
    """
    Checks if a dataset exists in the datasets directory.
    :param dataset_name: a string representing the name of the dataset
    :return: True if the dataset exists, False otherwise
    """

    dataset = os.path.join(datasets_dir, dataset_name)
    return os.path.isdir(dataset)


def download_and_store(dataset_name: str) -> None:
    """
    Downloads a dataset from the BEIR datasets repository and stores it in the datasets directory.
    :param dataset_name: a string representing the name of the dataset to download
    """

    url = BEIR_URL.format(dataset_name)
    util.download_and_unzip(url, datasets_dir)



def get_dataset(dataset_name:str) -> None:
    """
    Downloads a dataset from the BEIR datasets repository.
    :param dataset_name: a string representing the name of the dataset to download
    """

    if does_exist(dataset_name):
        print("Dataset already exists")

    else:
        download_and_store(dataset_name)
        print(f"Dataset {dataset_name} downloaded")