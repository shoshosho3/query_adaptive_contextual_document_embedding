from torch.utils.data import DataLoader
from models.more_positive import *
from typing import Tuple

def initial_prints(model):
    print("*" * 40 + f" {model} " + "*" * 40)
    print("Training the model...")
    print()


def _prepare_single_embeddings_training(doc_embeddings: torch.Tensor, train_query_embeddings:torch.Tensor,
                                        train_queries:list, train_qrels: dict, corpus: dict) -> Tuple[QueryDataset, DataLoader]:
    """
    Helper function to prepare single embeddings dataset and dataloader

    Args:
        doc_embeddings: Tensor of document embeddings by CDE method.
        train_query_embeddings: Tensor of document embeddings by CDE method.
        train_queries: Dictionary of the train queries.
        train_qrels: Dictionary of the relevance of the documents for the train queries.
        corpus: Dictionary that contains all the documents of the corpus.

    Returns:
          A tuple that contains the dataset and the dataloader.
    """
    dataset = QueryDataset(doc_embeddings, train_query_embeddings, list(train_queries.keys()),train_qrels,
                           list(corpus.keys()), num_negatives=500, max_positives=5)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

    return dataset, dataloader


def _prepare_multi_embeddings_training(doc_embeddings: torch.Tensor, train_query_embeddings: torch.Tensor,
                                       train_query_embeddings_bert: torch.Tensor, train_query_embeddings_tfidf: torch.Tensor,
                                       train_queries: dict, train_qrels: dict, corpus: dict) -> Tuple[MultiEmbeddingsQueryDataset,
                                                                                                      DataLoader]:
    """
    Helper function to prepare single embeddings dataset and dataloader

    Args:
        doc_embeddings: Tensor of document embeddings by CDE method.
        train_query_embeddings: Tensor of document embeddings by CDE method.
        train_query_embeddings_bert: Tensor of document embeddings by BERT method.
        train_query_embeddings_tfidf: Tensor of document embeddings by TF-IDF method.
        train_queries: Dictionary of the train queries.
        train_qrels: Dictionary of the relevance of the documents for the train queries.
        corpus: Dictionary that contains all the documents of the corpus.

    Returns:
           A tuple that contains the dataset and the dataloader.
    """
    dataset = MultiEmbeddingsQueryDataset(doc_embeddings, train_query_embeddings, train_query_embeddings_bert,
                                          train_query_embeddings_tfidf, list(train_queries.keys()), train_qrels,
                                          list(corpus.keys()), num_negatives=500, max_positives=5)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=multi_custom_collate_fn)
    return dataset, dataloader


def _train_single_embeddings_model(adaptive_model: torch.nn.Module, dataloader: DataLoader, criterion: torch.nn.Module,
                                   optimizer: torch.optim.Optimizer) -> None:
    """
    Helper function to train single embeddings model.

    Args:
        adaptive_model: The Query Adaptive Model.
        dataloader: The DataLoader.
        criterion: The Loss Function.
        optimizer: The Optimizer.
    """
    train_query_adaptive_model(adaptive_model, dataloader, criterion, optimizer, num_epochs=1)


def _train_multi_embeddings_model(adaptive_model: torch.nn.Module, dataloader: DataLoader, criterion: torch.nn.Module,
                                  optimizer: torch.optim.Optimizer) -> None:
    """
        Helper function to train single embeddings model.

        Args:
            adaptive_model: The Multi-Embeddings-Query Adaptive Model.
            dataloader: The DataLoader used to load the training examples.
            criterion: The Loss Function used to compute training loss.
            optimizer: The Optimizer used to update weights.
    """
    train_multi_embeddings_query_adaptive_model(adaptive_model, dataloader, criterion, optimizer, num_epochs=1)


def _evaluate_single_embeddings_model(adaptive_model: torch.nn.Module, doc_embeddings: torch.Tensor,
                                      test_query_embeddings: torch.Tensor, test_queries: dict, test_qrels: dict,
                                      corpus: dict, calculate_map: callable) -> float:
    """
    Helper function to evaluate multi embeddings model

    Args:
        adaptive_model: The Multi-Embeddings-Query Adaptive Model.
        doc_embeddings: Tensor of the document embeddings by CDE method.
        test_query_embeddings: Tensor of the query embeddings by CDE method.
        test_queries: Dictionary of all test queries.
        test_qrels: Dictionary of the relevant documents for the queries.
        corpus: Dictionary of all the documents of the corpus.

    Returns:
            The Mean Average Precision (MAP) of the model on the test set.
    """


    return calculate_map(adaptive_model, doc_embeddings, test_query_embeddings, list(test_queries.keys()), test_qrels,
                         list(corpus.keys()))


def _evaluate_multi_embeddings_model(adaptive_model: torch.nn.Module, doc_embeddings: torch.Tensor,
                                     test_query_embeddings: torch.Tensor,test_query_embeddings_bert: torch.Tensor,
                                     test_query_embeddings_tfidf: torch.Tensor, test_queries: dict, test_qrels: dict,
                                     corpus: dict, calculate_map: callable) -> float:
    """
    Helper function to evaluate multi embeddings model

    Args:
        adaptive_model: The Multi-Embeddings-Query Adaptive Model.
        doc_embeddings: Tensor of the document embeddings by CDE method.
        test_query_embeddings: Tensor of the query embeddings by CDE method.
        test_query_embeddings_bert: Tensor of the query embeddings by BERT method.
        test_query_embeddings_tfidf: Tensor of the query embeddings by TF-IDF method.
        test_queries: Dictionary of all test queries.
        test_qrels: Dictionary of the relevant documents for the queries.
        corpus: Dictionary of all the documents of the corpus.

    Returns:
          The Mean Average Precision (MAP) of the model on the test set.
    """

    return calculate_map(adaptive_model, doc_embeddings, test_query_embeddings, test_query_embeddings_bert,
                         test_query_embeddings_tfidf, list(test_queries.keys()), test_qrels, list(corpus.keys()))


def train_adaptive_cde(doc_embeddings: torch.Tensor, train_query_embeddings: torch.Tensor,
                       test_query_embeddings: torch.Tensor, train_queries: dict, test_queries: dict, train_qrels: dict,
                       test_qrels: dict, corpus: dict, num_epochs: int, adaptive_model: torch.nn.Module,
                       criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, dataset_name: str,
                       calculate_map: callable, save_model: callable, model_name: str="QACDE",
                       is_multi_embeddings: bool=False, train_query_embeddings_bert: torch.Tensor=None,
                       train_query_embeddings_tfidf: torch.Tensor=None, test_query_embeddings_bert: torch.Tensor=None,
                       test_query_embeddings_tfidf: torch.Tensor=None) -> None:
    """
    Trains an adaptive CDE model using document embeddings and query embeddings (either single or multi-embeddings).
    The model is evaluated using Mean Average Precision (MAP) on a test set, and the best model is saved.

    Args:
        doc_embeddings: Tensor of document embeddings by CDE method of shape (num_docs, embedding_dim).
        train_query_embeddings: Tensor of training query embeddings by CDE method of shape (num_queries, embedding_dim).
        test_query_embeddings: Tensor of test query embeddings by CDE method of shape (num_queries, embedding_dim).
        train_queries: Dictionary of training queries (query_id: query_text).
        test_queries: Dictionary of test queries (query_id: query_text).
        train_qrels: Dictionary containing training query relevance labels (query_id: {doc_id1, doc_id2, ...}).
        test_qrels: Dictionary containing test query relevance labels (query_id: {doc_id1, doc_id2, ...}).
        corpus: Dictionary containing the document corpus (doc_id: doc_text).
        num_epochs: Number of epochs for training.
        adaptive_model: The (multi-embeddings)-query-adaptive model to be trained.
        criterion: The loss function used to compute training loss.
        optimizer: The optimizer used to update model weights.
        dataset_name: The name of the dataset being used.
        calculate_map: A function to calculate the Mean Average Precision (MAP) score.
        save_model: A function to save the model.
        model_name: Name of the model for saving purposes.
        is_multi_embeddings: Whether to use multiple query embeddings (BERT, TF-IDF, etc.).
        train_query_embeddings_bert: Tensor of training query embeddings by BERT method of shape (num_queries, embedding_dim).
        train_query_embeddings_tfidf: Tensor of training query embeddings by TF-IDF method of shape (num_queries, embedding_dim).
        test_query_embeddings_bert: Tensor of testing query embeddings by BERT method of shape (num_queries, embedding_dim).
        test_query_embeddings_tfidf: Tensor of testing query embeddings by TF-IDF method of shape (num_queries, embedding_dim).
    """
    model_type = 'Multi Embeddings Query Adaptive CDE' if is_multi_embeddings else 'Query Adaptive CDE'
    initial_prints(model_type)

    # initializations
    best_map_score = 0.0

    # loop over the epochs
    for i in range(num_epochs):
        print(f"Epoch {i + 1}:")

        # Prepare dataset and dataloader based on model type
        if is_multi_embeddings:
            _, dataloader = _prepare_multi_embeddings_training(doc_embeddings, train_query_embeddings,
                                                               train_query_embeddings_bert, train_query_embeddings_tfidf,
                                                               train_queries, train_qrels, corpus)

            # Train the model
            _train_multi_embeddings_model(adaptive_model, dataloader, criterion, optimizer)

            # Evaluate the model
            test_map_score = _evaluate_multi_embeddings_model(adaptive_model, doc_embeddings, test_query_embeddings,
                                                              test_query_embeddings_bert, test_query_embeddings_tfidf,
                                                              test_queries, test_qrels, corpus, calculate_map)
        else:
            _, dataloader = _prepare_single_embeddings_training(doc_embeddings, train_query_embeddings, train_queries,
                                                                train_qrels, corpus)

            # Train the model
            # _train_single_embeddings_model(adaptive_model, dataloader, criterion, optimizer)

            # Evaluate the model
            test_map_score = _evaluate_single_embeddings_model(adaptive_model, doc_embeddings, test_query_embeddings,
                                                               test_queries, test_qrels, corpus, calculate_map)

        # Save the model if the MAP score is better
        if test_map_score >= best_map_score:
            save_model(adaptive_model, model_name, dataset_name)
            best_map_score = test_map_score

        # Print the MAP score
        print(f"Mean Average Precision (MAP) On Test Set: {test_map_score:.4f}\n")

    print('*' * 40)
    print('End of training, best MAP score:', best_map_score)
    print()