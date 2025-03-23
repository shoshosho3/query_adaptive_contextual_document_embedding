import torch
from beir.datasets.data_loader import GenericDataLoader
from models.with_attention import QueryAdaptiveCDE, MultiEmbeddingsQueryAdaptiveCDE
from analysis.results_utils import evaluate_models, calculate_map, calculate_multi_map
import models.more_positive as more_positive
from utils.param_utils import get_args
from pickle_utils import open_pickles
from consts import *
from training.training_query_adaptive_layer import train_adaptive_cde
from pre_trained_cde.save_pre_trained import save_model


if __name__ == "__main__":

    # getting the arguments
    args = get_args(for_query_adaptive_cde=False)

    # getting the data path
    data_path = "datasets/" + args.dataset

    # getting the data
    corpus, queries_train, qrels_train = GenericDataLoader(data_path).load(split="train")
    _, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")
    corpus_list = list(corpus.items())

    # setting the device - not used at the moment!!!!!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # evaluating the models
    evaluate_models(args.dataset, doc_embeddings_tensor, test_query_embeddings_tensor,
                    test_query_embeddings_tensor_bert, test_query_embeddings_tensor_tfidf, test_queries, test_qrels,
                    corpus, MultiEmbeddingsQueryAdaptiveCDE)
