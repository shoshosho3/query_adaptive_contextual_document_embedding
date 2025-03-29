import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from models.with_attention import QueryAdaptiveCDE, MultiEmbeddingsQueryAdaptiveCDE
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import pickle

def evaluate_models(dataset_name: str, doc_embeddings: torch.Tensor, test_query_embeddings: torch.Tensor,
                    test_query_embeddings_bert: torch.Tensor, test_query_embeddings_tfidf: torch.Tensor,
                    test_queries: list, test_qrels: dict, corpus: dict) -> None:

    """
    Loads and evaluates our query adaptive models.

    Args:
        dataset_name: Name of the dataset.
        doc_embeddings: Tensor of document embeddings by CDE method of shape (num_docs, embedding_dim).
        test_query_embeddings: Tensor of test query embeddings by CDE method of shape(num_queries, embedding_dim).
        test_query_embeddings_bert: Tensor of test query embeddings by BERT method of shape (num_queries, embedding_dim).
        test_query_embeddings_tfidf: Tensor of test query embeddings by TF-IDF method of shape (num_queries, embeddings_dim).
        test_queries: List of query IDs.
        test_qrels: Dictionary of shape {query_id: {relevant_doc_id1, relevant_doc_id2, ...}}.
        corpus: Dictionary of shape {doc_id: {title: doc_title, text: doc_text}}.
    """

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


def evaluate_baseline(dataset_name: str, doc_embeddings: torch.Tensor, test_query_embeddings: torch, test_queries: list,
                      test_qrels: dict, corpus: dict) -> None:
    """
    Calculates and prints the Mean Average Precision (MAP) of the baseline model.

    Args:
        dataset_name: Name of the dataset.
        doc_embeddings: Tensor of document embeddings by CDE method of shape (num_docs, embedding_dim).
        test_query_embeddings: Tensor of test query embeddings by CDE method of shape(num_queries, embedding_dim).
        test_queries: List of query IDs.
        test_qrels: Dictionary of shape {query_id: {relevant_doc_id1, relevant_doc_id2, ...}}.
        corpus: Dictionary of shape {doc_id: {title: doc_title, text: doc_text}}.
    """

    map_score = calculate_map(None, doc_embeddings, test_query_embeddings,
                              list(test_queries.keys()), test_qrels, list(corpus.keys()))

    print(f"MAP Score for the CDE Model on the Test Set of {dataset_name} is: {map_score}")


# def calculate_map(model: QueryAdaptiveCDE, document_embeddings: torch.Tensor, query_embeddings:torch.Tensor,
#                   queries: list, qrels: dict, doc_ids: list) -> float:
#     """
#     Calculate Mean Average Precision (MAP) for the query-adaptive and the baseline model.
#
#     Args:
#         model: The trained query adaptive model.
#         document_embeddings: Tensor of document embeddings by CDE method of shape (num_docs, embedding_dim).
#         query_embeddings: Tensor of query embeddings by CDE method of shape (num_queries, embedding_dim).
#         queries: List of query IDs.
#         qrels: Dictionary of shape {query_id: {relevant_doc_id1, relevant_doc_id2, ...}}.
#         doc_ids: List of document IDs corresponding to document_embeddings.
#
#     Returns:
#         MAP score.
#     """
#     if model:
#         model.eval()
#
#         device = next(model.parameters()).device
#     else:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     document_embeddings = document_embeddings.to(device)
#     query_embeddings = query_embeddings.to(device)
#
#     map_score = 0.0
#
#     doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
#
#     for query_idx, query_id in tqdm(enumerate(queries)):
#         query_embedding = query_embeddings[query_idx].to(device)
#
#         with torch.no_grad():
#             batch_size = 1024
#             similarity_scores = []
#
#             for i in range(0, len(document_embeddings), batch_size):
#                 batch_docs = document_embeddings[i:i + batch_size]
#                 batch_query = query_embedding.unsqueeze(0).repeat(len(batch_docs), 1)
#
#                 if model:
#                     adaptive_doc_embeddings = model(batch_docs, batch_query)
#                 else:
#                     adaptive_doc_embeddings = batch_docs
#
#                 batch_scores = F.cosine_similarity(query_embedding.unsqueeze(0), adaptive_doc_embeddings, dim=1)
#                 similarity_scores.append(batch_scores)
#
#             similarity_scores = torch.cat(similarity_scores)
#
#         relevant_doc_ids = qrels.get(query_id, set())
#         relevant_doc_indices = [doc_id_to_idx[doc_id] for doc_id in relevant_doc_ids if doc_id in doc_id_to_idx]
#
#         y_true = np.zeros(len(doc_ids))
#         y_true[relevant_doc_indices] = 1
#         y_score = similarity_scores.cpu().numpy()
#
#         ap = average_precision_score(y_true, y_score)
#         map_score += ap
#
#     map_score /= len(queries)
#     return map_score

def calculate_map(
    model: QueryAdaptiveCDE,
    document_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    queries: list,
    qrels: dict,
    doc_ids: list,
    batch_size: int = 128  # Process multiple queries at once
) -> float:
    """
    Optimized function to calculate Mean Average Precision (MAP).
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

    # Process queries in batches
    num_queries = len(queries)
    for start in tqdm(range(0, num_queries, batch_size), desc="Processing Queries"):
        end = min(start + batch_size, num_queries)
        batch_query_ids = queries[start:end]
        batch_query_embeddings = query_embeddings[start:end]

        with torch.no_grad():
            if model:
                batch_query_embeddings = batch_query_embeddings.unsqueeze(1)  # Add singleton for broadcasting
                document_embeddings_expanded = document_embeddings.unsqueeze(0).expand(batch_query_embeddings.shape[0], -1, -1)
                batch_query_embeddings = batch_query_embeddings.expand(-1, document_embeddings.shape[0], -1)
                adaptive_doc_embeddings = model(document_embeddings_expanded, batch_query_embeddings)
            else:
                adaptive_doc_embeddings = document_embeddings

            # Compute cosine similarity in a single operation
            similarity_scores = F.cosine_similarity(
                batch_query_embeddings.unsqueeze(2),  # (batch_size, num_docs, embedding_dim, 1)
                adaptive_doc_embeddings.unsqueeze(1),  # (batch_size, 1, num_docs, embedding_dim)
                dim=-1
            ).squeeze()

            similarity_scores = similarity_scores.cpu().numpy()

        # Compute AP for each query
        for i, query_id in enumerate(batch_query_ids):
            relevant_doc_ids = qrels.get(query_id, set())
            relevant_doc_indices = [doc_id_to_idx[doc_id] for doc_id in relevant_doc_ids if doc_id in doc_id_to_idx]

            y_true = np.zeros(len(doc_ids))
            y_true[relevant_doc_indices] = 1
            y_score = similarity_scores[i]

            map_score += average_precision_score(y_true, y_score)

    map_score /= num_queries
    return map_score


def calculate_multi_map(model: MultiEmbeddingsQueryAdaptiveCDE, document_embeddings: torch.Tensor,
                        query_embeddings: torch.Tensor, query_embeddings_bert: torch.Tensor, query_embeddings_tfidf:torch.Tensor,
                        queries: list, qrels: dict, doc_ids: list) -> float:
    """
        Calculate Mean Average Precision (MAP) for the multi-embeddings-query adaptive model.

        Args:
            model: Trained multi-embeddings-query-adaptive model.
            document_embeddings: Tensor of document embeddings by CDE method of shape (num_docs, embedding_dim).
            query_embeddings: Tensor of query embeddings by the CDE method of shape (num_queries, embedding_dim).
            query_embeddings_bert: Tensor of query embeddings by the BERT method of shape (num_queries, embedding_dim).
            query_embeddings_tfidf: Tensor of query embeddings by the TF-IDF method of shape (num_queries, embedding_dim).
            queries: List of query IDs.
            qrels: Dictionary of shape {query_id: {relevant_doc_id1, relevant_doc_id2, ...}}.
            doc_ids: List of document IDs corresponding to document_embeddings.

        Returns:
            MAP score.
        """

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
