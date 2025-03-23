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
    args = get_args()
    torch.manual_seed(args.seed)

    # getting the data path
    data_path = "datasets/" + args.dataset

    # getting the data
    corpus, queries_train, qrels_train = GenericDataLoader(data_path).load(split="train")
    _, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")
    corpus_list = list(corpus.items())

    # getting the different embeddings
    doc_embeddings_tensor, train_query_embeddings_tensor, train_query_embeddings_tensor_bert, \
    train_query_embeddings_tensor_tfidf, test_query_embeddings_tensor, test_query_embeddings_tensor_bert, \
    test_query_embeddings_tensor_tfidf = open_pickles(args)

    # setting the device - not used at the moment!!!!!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # getting parameters
    epochs = args.epochs
    hidden_dim = args.hidden_dim

    # setting critertion for the loss
    criterion = more_positive.MultiPositiveLoss()

    # setting the models
    query_adaptive_model = QueryAdaptiveCDE(EMBEDDINGS_DIMENSION, hidden_dim, num_heads=4)
    multi_embeddings_query_adaptive_model = MultiEmbeddingsQueryAdaptiveCDE(EMBEDDINGS_DIMENSION,
                                                                            hidden_dim,
                                                                            train_query_embeddings_tensor_tfidf,
                                                                            num_heads=4)

    # setting the optimizers
    optimizer_query_adaptive = torch.optim.Adam(query_adaptive_model.parameters(), lr=1e-5)
    optimizer_multi_embeddings_query_adaptive = torch.optim.Adam(multi_embeddings_query_adaptive_model.parameters(),
                                                                 lr=1e-5)

    # training the query adaptive model
    train_adaptive_cde(doc_embeddings_tensor,
                       train_query_embeddings_tensor,
                       test_query_embeddings_tensor,
                       queries_train,
                       test_queries,
                       qrels_train,
                       test_qrels,
                       corpus,
                       epochs,
                       query_adaptive_model,
                       criterion,
                       optimizer_query_adaptive,
                       args.dataset,
                       more_positive,
                       calculate_map,
                       save_model)

    # training the multi embeddings model
    train_adaptive_cde(
        doc_embeddings=doc_embeddings_tensor,
        train_query_embeddings=train_query_embeddings_tensor,
        test_query_embeddings=test_query_embeddings_tensor,
        train_queries=queries_train,
        test_queries=test_queries,
        train_qrels=qrels_train,
        test_qrels=test_qrels,
        corpus=corpus,
        num_epochs=epochs,
        adaptive_model=multi_embeddings_query_adaptive_model,
        criterion=criterion,
        optimizer=optimizer_multi_embeddings_query_adaptive,
        dataset_name=args.dataset,
        more_positive=more_positive,
        calculate_map=calculate_multi_map,
        save_model=save_model,
        model_name="MEQACDE",
        is_multi_embeddings=True,
        train_query_embeddings_bert=train_query_embeddings_tensor_bert,
        train_query_embeddings_tfidf=train_query_embeddings_tensor_tfidf,
        test_query_embeddings_bert=test_query_embeddings_tensor_bert,
        test_query_embeddings_tfidf=test_query_embeddings_tensor_tfidf
    )


    # evaluating the models
    evaluate_models(args.dataset, doc_embeddings_tensor, test_query_embeddings_tensor,
                    test_query_embeddings_tensor_bert, test_query_embeddings_tensor_tfidf, test_queries, test_qrels,
                    corpus, MultiEmbeddingsQueryAdaptiveCDE)
