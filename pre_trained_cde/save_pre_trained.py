import pickle
import torch
import os


def check_directory(directory):

    if not os.path.isdir(directory):
        os.makedirs(directory)

def save_pickles(docs, train_queries, dev_queries, test_queries, dataset_name, index):
    """
    Saves embeddings for documents and queries into pickle files.

    Args:
        docs (torch.Tensor): The tensor of document embeddings.
        train_queries (torch.Tensor): The tensor of training query embeddings.
        dev_queries (torch.Tensor): The tensor of validation/development query embeddings.
        test_queries (torch.Tensor): The tensor of test query embeddings.
        dataset_name (str): The name of the dataset being processed.
        index (int): The current index, used in the filenames to organize data versions.

    Saves:
        Four pickle files for document, training, validation, and test embeddings named
        using the dataset name and index.
    """

    check_directory(dataset_name)

    with open(f'{dataset_name}/doc_embeddings_{dataset_name}_{index}.pkl', 'wb') as f:
        pickle.dump(docs, f)

    with open(f'{dataset_name}/query_embeddings_train_{dataset_name}_{index}.pkl', 'wb') as f:
        pickle.dump(train_queries, f)

    if dev_queries is not None:
        with open(f'{dataset_name}/query_embeddings_dev_{dataset_name}_{index}.pkl', 'wb') as f:
            pickle.dump(dev_queries, f)

    if test_queries is not None:
        with open(f'{dataset_name}/query_embeddings_test_{dataset_name}_{index}.pkl', 'wb') as f:
            pickle.dump(test_queries, f)


def normalize(docs):
    """
    Normalizes the given embeddings (tensors) using L2 normalization.

    Args:
        docs (list[torch.Tensor]): A list of tensors representing embeddings.

    Returns:
        torch.Tensor: A single tensor where each row is L2-normalized.
    """
    docs_tensor = torch.cat(docs, dim=0)
    docs_tensor /= docs_tensor.norm(p=2, dim=1, keepdim=True)
    return docs_tensor


def find_last_index(dataset_name):
    """
    Finds the next available index for saving embeddings for the given dataset.

    Args:
        dataset_name (str): The name of the dataset being processed.

    Returns:
        int: The next index to use for naming new embeddings files.
    """
    index = 1
    while True:
        try:
            with open(f'{dataset_name}/doc_embeddings_{dataset_name}_{index}.pkl', 'rb') as f:
                index += 1
        except FileNotFoundError:
            return index


def save(docs, train_queries, dev_queries, test_queries, dataset_name):
    """
    Normalizes and saves embeddings for documents and queries into versioned pickle files.

    Args:
        docs (list[torch.Tensor]): A list of document embedding tensors.
        train_queries (list[torch.Tensor]): A list of training query embedding tensors.
        dev_queries (list[torch.Tensor]): A list of validation/dev query embedding tensors.
        test_queries (list[torch.Tensor]): A list of test query embedding tensors.
        dataset_name (str): The name of the dataset being processed.

    Steps:
        1. Embeddings are normalized using L2 normalization.
        2. The last available index for filenames is discovered.
        3. Normalized embeddings are saved into pickle files with the appropriate names.
    """
    # Normalize embeddings
    docs_tensor = normalize(docs)
    train_query_tensor = normalize(train_queries)
    dev_query_tensor = normalize(dev_queries) if dev_queries else None
    test_query_tensor = normalize(test_queries) if test_queries else None

    # Find last index
    index = find_last_index(dataset_name)

    # Save embeddings
    save_pickles(docs_tensor, train_query_tensor, dev_query_tensor, test_query_tensor, dataset_name, index)
