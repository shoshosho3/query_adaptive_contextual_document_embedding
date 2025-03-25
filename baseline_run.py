import torch
from beir.datasets.data_loader import GenericDataLoader
from analysis.results_utils import evaluate_baseline
from utils.param_utils import get_args
from pickle_utils import open_pickles


if __name__ == "__main__":
    # getting the arguments
    args = get_args(for_query_adaptive_cde=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # getting the data path
    data_path = "datasets/" + args.dataset
    torch.manual_seed(args.seed)

    # getting the data
    corpus, queries_train, qrels_train = GenericDataLoader(data_path).load(split="train")
    _, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")
    corpus_list = list(corpus.items())


    # getting the different embeddings
    doc_embeddings_tensor, train_query_embeddings_tensor, train_query_embeddings_tensor_bert, \
        train_query_embeddings_tensor_tfidf, test_query_embeddings_tensor, test_query_embeddings_tensor_bert, \
        test_query_embeddings_tensor_tfidf = open_pickles(args, for_query_adaptive_cde=False)

    evaluate_baseline(args.dataset, doc_embeddings_tensor, test_query_embeddings_tensor, test_queries, test_qrels,
                      corpus)
