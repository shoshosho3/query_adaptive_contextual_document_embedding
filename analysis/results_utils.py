import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import pickle

def evaluate_models(dataset_name, doc_embeddings, test_query_embeddings, test_query_embeddings_bert,
                    test_query_embeddings_tfidf, test_queries, test_qrels, corpus, MultiEmbeddingsQueryAdaptiveCDE):

    print("Evaluating the models...")

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


def evaluate_baseline(dataset_name, doc_embeddings, test_query_embeddings, test_queries, test_qrels, corpus):

    map_score = calculate_map(None, doc_embeddings, test_query_embeddings,
                              list(test_queries.keys()), test_qrels, list(corpus.keys()))

    print(f"MAP Score for the CDE Model on the Test Set of {dataset_name} is: {map_score}")


def calculate_map(model, document_embeddings, query_embeddings, queries, qrels, doc_ids):
    """
    Calculate Mean Average Precision (MAP) for the query-adaptive model.

    Args:
        model: Trained query-adaptive model.
        document_embeddings: Tensor of shape (num_docs, embedding_dim).
        query_embeddings: Tensor of shape (num_queries, embedding_dim).
        queries: List of query IDs.
        qrels: Dictionary of shape {query_id: {relevant_doc_id1, relevant_doc_id2, ...}}.
        doc_ids: List of document IDs corresponding to document_embeddings.

    Returns:
        MAP score.
    """
    if model:
        model.eval()

        device = next(model.parameters()).device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    document_embeddings = document_embeddings.to(device)
    query_embeddings = query_embeddings.to(device)

    map_score = 0.0

    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    for query_idx, query_id in tqdm(enumerate(queries)):
        query_embedding = query_embeddings[query_idx].to(device)

        with torch.no_grad():
            batch_size = 1024
            similarity_scores = []

            for i in range(0, len(document_embeddings), batch_size):
                batch_docs = document_embeddings[i:i + batch_size]
                batch_query = query_embedding.unsqueeze(0).repeat(len(batch_docs), 1)

                if model:
                    adaptive_doc_embeddings = model(batch_docs, batch_query)
                else:
                    adaptive_doc_embeddings = batch_docs

                batch_scores = F.cosine_similarity(query_embedding.unsqueeze(0), adaptive_doc_embeddings, dim=1)
                similarity_scores.append(batch_scores)

            similarity_scores = torch.cat(similarity_scores)

        relevant_doc_ids = qrels.get(query_id, set())
        relevant_doc_indices = [doc_id_to_idx[doc_id] for doc_id in relevant_doc_ids if doc_id in doc_id_to_idx]

        y_true = np.zeros(len(doc_ids))
        y_true[relevant_doc_indices] = 1
        y_score = similarity_scores.cpu().numpy()

        ap = average_precision_score(y_true, y_score)
        map_score += ap

    map_score /= len(queries)
    return map_score


def calculate_multi_map(model, document_embeddings, query_embeddings, query_embeddings_bert, query_embeddings_tfidf,
                        queries, qrels, doc_ids):
    model.eval()

    device = next(model.parameters()).device
    document_embeddings = document_embeddings.to(device)
    query_embeddings = query_embeddings.to(device)
    query_embeddings_bert = query_embeddings_bert.to(device)
    query_embeddings_tfidf = query_embeddings_tfidf.to(device)

    map_score = 0.0

    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    for query_idx, query_id in tqdm(enumerate(queries)):
        query_embedding = query_embeddings[query_idx].to(device)
        query_embedding_bert = query_embeddings_bert[query_idx].to(device)
        query_embedding_tfidf = query_embeddings_tfidf[query_idx].to(device)

        with torch.no_grad():
            batch_size = 1024
            similarity_scores = []

            for i in range(0, len(document_embeddings), batch_size):
                batch_docs = document_embeddings[i:i + batch_size]
                batch_query = query_embedding.unsqueeze(0).repeat(len(batch_docs), 1)
                batch_query_bert = query_embedding_bert.unsqueeze(0).repeat(len(batch_docs), 1)
                batch_query_tfidf = query_embedding_tfidf.unsqueeze(0).repeat(len(batch_docs), 1)

                multi_q_emb_adaptive_doc_embeddings = model(batch_docs, batch_query, batch_query_bert,
                                                            batch_query_tfidf)
                batch_scores = F.cosine_similarity(query_embedding.unsqueeze(0), multi_q_emb_adaptive_doc_embeddings,
                                                   dim=1)
                similarity_scores.append(batch_scores)

            similarity_scores = torch.cat(similarity_scores)

        relevant_doc_ids = qrels.get(query_id, set())
        relevant_doc_indices = [doc_id_to_idx[doc_id] for doc_id in relevant_doc_ids if doc_id in doc_id_to_idx]

        y_true = np.zeros(len(doc_ids))
        y_true[relevant_doc_indices] = 1
        y_score = similarity_scores.cpu().numpy()

        ap = average_precision_score(y_true, y_score)
        map_score += ap

    map_score /= len(queries)
    return map_score
