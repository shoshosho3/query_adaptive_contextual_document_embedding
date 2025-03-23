import torch
from beir.datasets.data_loader import GenericDataLoader
from analysis.results_utils import evaluate_baseline
from utils.param_utils import get_args
from pickle_utils import open_pickles


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

    # getting the different embeddings
    doc_embeddings_tensor, train_query_embeddings_tensor, train_query_embeddings_tensor_bert, \
        train_query_embeddings_tensor_tfidf, test_query_embeddings_tensor, test_query_embeddings_tensor_bert, \
        test_query_embeddings_tensor_tfidf = open_pickles(args)

    evaluate_baseline(args.dataset, doc_embeddings_tensor, test_query_embeddings_tensor, test_queries, test_qrels,
                      corpus)
