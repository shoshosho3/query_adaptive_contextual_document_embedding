import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

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
    # Ensure model is in eval mode
    model.eval()

    # Move data to the same device as model
    device = next(model.parameters()).device
    document_embeddings = document_embeddings.to(device)
    query_embeddings = query_embeddings.to(device)

    map_score = 0.0

    # Create a mapping from doc_id to index
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    for query_idx, query_id in tqdm(enumerate(queries)):
        query_embedding = query_embeddings[query_idx].to(device)

        # Compute similarity scores between the query and all documents
        with torch.no_grad():
            # Process in batches if document set is large
            batch_size = 1024  # Adjust based on your GPU memory
            similarity_scores = []

            # Get adaptive query embedding once
            # adaptive_query_embedding = model(query_embedding.unsqueeze(0), query_embedding.unsqueeze(0))

            # Process documents in batches
            for i in range(0, len(document_embeddings), batch_size):
                batch_docs = document_embeddings[i:i + batch_size]
                batch_query = query_embedding.unsqueeze(0).repeat(len(batch_docs), 1)

                adaptive_doc_embeddings = model(batch_docs, batch_query)
                batch_scores = F.cosine_similarity(query_embedding.unsqueeze(0), adaptive_doc_embeddings, dim=1)
                similarity_scores.append(batch_scores)

            # Concatenate all batch results
            similarity_scores = torch.cat(similarity_scores)

        # Move rankings to CPU for sklearn metrics
        ranked_doc_indices = torch.argsort(similarity_scores, descending=True).cpu().numpy()

        # Get relevant document indices for this query
        relevant_doc_ids = qrels.get(query_id, set())
        relevant_doc_indices = [doc_id_to_idx[doc_id] for doc_id in relevant_doc_ids if doc_id in doc_id_to_idx]

        # Convert ranked list and relevant indices to binary relevance
        y_true = np.zeros(len(doc_ids))
        y_true[relevant_doc_indices] = 1
        y_score = similarity_scores.cpu().numpy()

        # Calculate Average Precision (AP) for this query
        ap = average_precision_score(y_true, y_score)
        map_score += ap

    # Calculate MAP as the mean of APs across all queries
    map_score /= len(queries)
    return map_score