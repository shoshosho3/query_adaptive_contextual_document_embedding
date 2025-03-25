import argparse
from beir.datasets.data_loader import GenericDataLoader
from query_embedders.query_embedders import QueryBertEmbedder, QueryTFIDFEmbedder, save_queries_embeddings_pickle
from warnings import filterwarnings
from consts import *

filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="BEIR dataset name.")
    parser.add_argument("--tfidf_max_dim", type=int, required=True, help="Max dimension of embedded documents with TF-IDF")

    args = parser.parse_args()
    dataset_name = args.dataset

    data_path = "datasets/" + args.dataset

    corpus, queries_train, qrels_train = GenericDataLoader(data_path).load(split="train")
    _, queries_test, qrels_test = GenericDataLoader(data_path).load(split="test")

    query_bert_embedder = QueryBertEmbedder(train_queries=queries_train, test_queries=queries_test,
                                            embedding_dim=EMBEDDINGS_DIMENSION)
    query_tfidf_embedder = QueryTFIDFEmbedder(corpus=list(corpus.values()), train_queries=queries_train,
                                              max_dim=args.tfidf_max_dim, test_queries=queries_test)

    save_queries_embeddings_pickle(dataset_name, [query_bert_embedder, query_tfidf_embedder], ["BERT", "TFIDF"])
