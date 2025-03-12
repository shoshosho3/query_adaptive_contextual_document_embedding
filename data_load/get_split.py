from beir.datasets.data_loader import GenericDataLoader
import os

datasets_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")
TSV = '.tsv'
QRELS_PATH = 'qrels'

CORPUS = 0
QUERIES = 1
QRELS = 2


def get_split(split_type, dataset_name):
    """
    This function loads a split of a dataset.
    :param split_type: train, dev, or test
    :param dataset_name: beir dataset name
    :return: the split of the dataset
    """


    dataset_path = os.path.join(datasets_dir, dataset_name)


    if os.path.isfile(os.path.join(dataset_path, QRELS_PATH, split_type + TSV)):
        return GenericDataLoader(dataset_path).load(split=split_type)

    else:
        return None



class GetSplit:
    """
    Manages train, dev, and test splits for a dataset.

    This class is designed to handle and provide access to different splits of
    a dataset. It ensures the existence of a train split and provides utility
    methods to verify the presence of dev and test splits. This class allows
    for uniform access to dataset splits for various applications.

    :ivar train: The train split of the dataset.
    :ivar dev: The dev/validation split of the dataset. Optional.
    :ivar test: The test split of the dataset. Optional.
    """

    def __init__(self, dataset_name):

        self.train = get_split("train", dataset_name)

        if self.train is None:
            ValueError("Dataset does not have a train set")

        self.dev = get_split("dev", dataset_name)
        self.test = get_split("test", dataset_name)

    def has_train(self):
        return self.train is not None

    def has_dev(self):
        return self.dev is not None

    def has_test(self):
        return self.test is not None
