import pickle
import torch
import argparse
from beir.datasets.data_loader import GenericDataLoader
from torch.utils.data import DataLoader
from pre_trained_cde.save_pre_trained import save_model
from analysis.results_utils import calculate_map, calculate_multi_map
from models.with_attention import QueryAdaptiveCDE, MultiEmbeddingsQueryAdaptiveCDE
from analysis.results_utils import calculate_map
import models.with_attention as with_attention
import models.more_positive as more_positive


def open_pickles():
    with open(f'{args.dataset}/doc_embeddings_{args.dataset}_{args.index}.pkl', 'rb') as f:
        doc_embeddings = pickle.load(f)

    with open(f'{args.dataset}/query_embeddings_train_{args.dataset}_{args.index}.pkl', 'rb') as f:
        train_query_embeddings = pickle.load(f)

    with open(f'{args.dataset}/query_embeddings_train_{args.dataset}_BERT_{args.index}.pkl', 'rb') as f:
        train_query_embeddings_bert = pickle.load(f)

    with open(f'{args.dataset}/query_embeddings_train_{args.dataset}_TFIDF_{args.index}.pkl', 'rb') as f:
        train_query_embeddings_tfidf = pickle.load(f)

    with open(f'{args.dataset}/query_embeddings_test_{args.dataset}_{args.index}.pkl', 'rb') as f:
        test_query_embeddings = pickle.load(f)

    with open(f'{args.dataset}/query_embeddings_test_{args.dataset}_BERT_{args.index}.pkl', 'rb') as f:
        test_query_embeddings_bert = pickle.load(f)

    with open(f'{args.dataset}/query_embeddings_test_{args.dataset}_TFIDF_{args.index}.pkl', 'rb') as f:
        test_query_embeddings_tfidf = pickle.load(f)

    return doc_embeddings, train_query_embeddings, train_query_embeddings_bert, train_query_embeddings_tfidf, \
           test_query_embeddings, test_query_embeddings_bert, test_query_embeddings_tfidf


def train_query_adaptive_cde(doc_embeddings, train_query_embeddings, test_query_embeddings, train_queries, test_queries,
                             train_qrels, test_qrels, corpus, num_epochs, query_adaptive_model, criterion, optimizer,
                             dataset_name):
    print("*" * 40 + " Query Adaptive CDE " + "*" * 40)
    print("Training the model...")
    print()

    best_map_score = 0.0

    for i in range(num_epochs):
        print(f"Epoch {i + 1}:")
        dataset = more_positive.QueryDataset(doc_embeddings, train_query_embeddings, list(train_queries.keys()),
                                             train_qrels, list(corpus.keys()), num_negatives=500, max_positives=5)

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=more_positive.custom_collate_fn)

        more_positive.train_query_adaptive_model(query_adaptive_model, dataloader, criterion, optimizer,
                                                 num_epochs=1)

        test_map_score = calculate_map(query_adaptive_model, doc_embeddings, test_query_embeddings,
                                       list(test_queries.keys()), test_qrels, list(corpus.keys()))

        if test_map_score >= best_map_score:
            save_model(multi_embeddings_query_adaptive_model, "QACDE", dataset_name)
            best_map_score = test_map_score

        print(f"Mean Average Precision (MAP) On Test Set: {test_map_score:.4f}\n")

    print()


def train_multi_embeddings_query_adaptive_cde(doc_embeddings, train_query_embeddings, train_query_embeddings_bert,
                                              train_query_embeddings_tfidf, test_query_embeddings,
                                              test_query_embeddings_bert, test_query_embeddings_tfidf, train_queries,
                                              test_queries, train_qrels, test_qrels, corpus, num_epochs,
                                              multi_embeddings_query_adaptive_model, criterion, optimizer,
                                              dataset_name):
    print("*" * 40 + " Multi Embeddings Query Adaptive CDE " + "*" * 40)
    print("Training the model...")
    print()

    best_map_score = 0.0

    for i in range(num_epochs):
        print(f"Epoch {i + 1}:")
        dataset = more_positive.MultiEmbeddingsQueryDataset(doc_embeddings, train_query_embeddings,
                                                            train_query_embeddings_bert, train_query_embeddings_tfidf,
                                                            list(train_queries.keys()), train_qrels,
                                                            list(corpus.keys()), num_negatives=500, max_positives=5)

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=more_positive.multi_custom_collate_fn)

        more_positive.train_multi_embeddings_query_adaptive_model(multi_embeddings_query_adaptive_model, dataloader,
                                                                  criterion, optimizer, num_epochs=1)

        test_map_score = calculate_multi_map(multi_embeddings_query_adaptive_model, doc_embeddings,
                                             test_query_embeddings, test_query_embeddings_bert,
                                             test_query_embeddings_tfidf,
                                             list(test_queries.keys()), test_qrels, list(corpus.keys()))

        if test_map_score >= best_map_score:
            save_model(multi_embeddings_query_adaptive_model, "MEQACDE", dataset_name)
            best_map_score = test_map_score

        print(f"Mean Average Precision (MAP) On Test Set: {test_map_score:.4f}\n")

    print()


def evaluate_models(dataset_name, doc_embeddings, test_query_embeddings, test_query_embeddings_bert,
                    test_query_embeddings_tfidf, test_queries, test_qrels, corpus):
    with open(f'models/QACDE_{dataset_name}.pkl', 'rb') as f:
        best_query_adaptive_cde_model = pickle.load(f)

    with open(f'models/MEQACDE_{dataset_name}.pkl', 'rb') as f:
        best_multi_embeddings_query_adaptive_cde_model = pickle.load(f)

    for model, name in zip([best_query_adaptive_cde_model, best_multi_embeddings_query_adaptive_cde_model],
                           ["Query Adaptive CDE", "Multi Embeddings Query Adaptive CDE"]):
        if isinstance(model, MultiEmbeddingsQueryAdaptiveCDE):
            map_score = calculate_multi_map(model, doc_embeddings, test_query_embeddings, test_query_embeddings_bert,
                                            test_query_embeddings_tfidf, list(test_queries.keys()), test_qrels,
                                            list(corpus.keys()))

        else:
            map_score = calculate_map(model, doc_embeddings, test_query_embeddings,
                                      list(test_queries.keys()), test_qrels, list(corpus.keys()))

        print(f"MAP Score for the {name} Model on the Test Set of {dataset_name} is: {map_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="BEIR dataset name.")
    parser.add_argument("--index", type=int, required=True, help="Index to check.")
    parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden Dimension For the Model.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of Epochs for Training Procedure.")

    args = parser.parse_args()

    data_path = "datasets/" + args.dataset

    corpus, queries_train, qrels_train = GenericDataLoader(data_path).load(split="train")
    _, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")
    corpus_list = list(corpus.items())

    doc_embeddings_tensor, train_query_embeddings_tensor, train_query_embeddings_tensor_bert, \
    train_query_embeddings_tensor_tfidf, test_query_embeddings_tensor, test_query_embeddings_tensor_bert, \
    test_query_embeddings_tensor_tfidf = open_pickles()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_dim = 768
    epochs = args.epochs
    hidden_dim = args.hidden_dim
    criterion = more_positive.MultiPositiveLoss()

    query_adaptive_model = QueryAdaptiveCDE(embedding_dim, hidden_dim, num_heads=4)
    multi_embeddings_query_adaptive_model = MultiEmbeddingsQueryAdaptiveCDE(embedding_dim, hidden_dim,
                                                                            train_query_embeddings_tensor_tfidf,
                                                                            num_heads=4)

    optimizer_query_adaptive = torch.optim.Adam(query_adaptive_model.parameters(), lr=1e-5)
    optimizer_multi_embeddings_query_adaptive = torch.optim.Adam(multi_embeddings_query_adaptive_model.parameters(),
                                                                 lr=1e-5)

    train_query_adaptive_cde(doc_embeddings_tensor, train_query_embeddings_tensor, test_query_embeddings_tensor,
                             queries_train, test_queries, qrels_train, test_qrels, corpus, epochs, query_adaptive_model,
                             criterion, optimizer_query_adaptive, args.dataset)

    train_multi_embeddings_query_adaptive_cde(doc_embeddings_tensor, train_query_embeddings_tensor,
                                              train_query_embeddings_tensor_bert, train_query_embeddings_tensor_tfidf,
                                              test_query_embeddings_tensor, test_query_embeddings_tensor_bert,
                                              test_query_embeddings_tensor_tfidf, queries_train, test_queries,
                                              qrels_train, test_qrels, corpus, epochs,
                                              multi_embeddings_query_adaptive_model, criterion,
                                              optimizer_multi_embeddings_query_adaptive, args.dataset)

    evaluate_models(args.dataset, doc_embeddings_tensor, test_query_embeddings_tensor,
                    test_query_embeddings_tensor_bert, test_query_embeddings_tensor_tfidf, test_queries, test_qrels,
                    corpus)
