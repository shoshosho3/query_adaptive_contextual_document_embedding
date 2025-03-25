import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from models.with_attention import QueryAdaptiveCDE, MultiEmbeddingsQueryAdaptiveCDE
from tqdm import tqdm
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QueryDataset(Dataset):
    """
    The class inherits the Dataset Class of PyTorch library adapting this for our specific task for Query Adaptive Model.
    """
    def __init__(self, document_embeddings: torch.Tensor, query_embeddings: torch.Tensor, queries: list, qrels: dict,
                 doc_ids: list, num_negatives: int=500, max_positives: int=50) -> None:
        """
        Initializes the QueryDataset Class.

        Args:
            document_embeddings: Tensor of the documents embeddings by CDE method of shape (num_docs, embedding_dim).
            query_embeddings: Tensor of the query embeddings by CDE method of shape (num_queries, embedding_dim).
            queries: List of the query IDs.
            qrels: Dictionary of shape {query_id: {relevant_doc_id1, relevant_doc_id2, ...}}.
            doc_ids: List of the documents IDs.
            num_negatives: Number of negative sampling examples.
            max_positives: Maximum number of positive examples.
        """
        super(QueryDataset).__init__()
        self.document_embeddings = document_embeddings.to(device)
        self.query_embeddings = query_embeddings.to(device)
        self.queries = queries
        self.qrels = qrels
        self.doc_ids = doc_ids
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        self.num_negatives = num_negatives
        self.max_positives = max_positives

        self.valid_queries = [q for q in queries if
                              q in qrels and any(doc_id in self.doc_id_to_idx for doc_id in qrels[q])]

    def __len__(self) -> int:
        """
        Returns the length of the dataset, which is the number of valid queries with at least one relevant document.
        """
        return len(self.valid_queries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Retrieves the query embedding, positive and negative document embeddings for a given index.

        Args:
            idx (int): The index of the query to retrieve.

        Returns:
            tuple: A tuple containing:
                - query_embedding (torch.Tensor): The embedding of the query.
                - pos_doc_embeddings (torch.Tensor): The embeddings of relevant (positive) documents.
                - neg_doc_embeddings (torch.Tensor): The embeddings of non-relevant (negative) documents.
                - num_positives (int): The number of positive documents.
        """

        query_id = self.valid_queries[idx]
        query_embedding = self.query_embeddings[self.queries.index(query_id)]

        relevant_doc_ids = self.qrels[query_id]
        relevant_doc_indices = [self.doc_id_to_idx[doc_id] for doc_id in relevant_doc_ids
                                if doc_id in self.doc_id_to_idx]

        if len(relevant_doc_indices) > self.max_positives:
            relevant_doc_indices = relevant_doc_indices[:self.max_positives]

        pos_doc_embeddings = self.document_embeddings[relevant_doc_indices]

        neg_doc_indices = []
        while len(neg_doc_indices) < self.num_negatives:
            neg_idx = torch.randint(0, len(self.doc_ids), (1,)).item()
            if neg_idx not in relevant_doc_indices and neg_idx not in neg_doc_indices:
                neg_doc_indices.append(neg_idx)

        neg_doc_embeddings = self.document_embeddings[neg_doc_indices]

        return query_embedding, pos_doc_embeddings, neg_doc_embeddings, len(relevant_doc_indices)


class MultiEmbeddingsQueryDataset(Dataset):
    """
    The class inherits the Dataset Class of PyTorch library adapting this for our specific task for Query Adaptive Model.
    """
    def __init__(self, document_embeddings: torch.Tensor, query_embeddings: torch.Tensor, query_embeddings_bert: torch.Tensor,
                 query_embeddings_tfidf: torch.Tensor, queries: list, qrels: dict, doc_ids: list, num_negatives: int=500,
                 max_positives: int=50) -> None:
        """
        Initializes the QueryDataset Class.

        Args:
            document_embeddings: Tensor of the documents embeddings by CDE method of shape (num_docs, embedding_dim).
            query_embeddings: Tensor of the query embeddings by CDE method of shape (num_queries, embedding_dim).
            query_embeddings_bert: Tensor of the query embeddings by BERT method of shape (num_queries, embedding_dim).
            query_embeddings_tfidf: Tensor of the query embeddings by TF-IDF method of shape (num_queries, embedding_dim).
            queries: List of the query IDs.
            qrels: Dictionary of shape {query_id: {relevant_doc_id1, relevant_doc_id2, ...}}.
            doc_ids: List of the documents IDs.
            num_negatives: Number of negative sampling examples.
            max_positives: Maximum number of positive examples.
        """
        super(MultiEmbeddingsQueryDataset).__init__()
        self.document_embeddings = document_embeddings.to(device)
        self.query_embeddings = query_embeddings.to(device)
        self.query_embeddings_bert = query_embeddings_bert.to(device)
        self.query_embeddings_tfidf = query_embeddings_tfidf.to(device)

        self.queries = queries
        self.qrels = qrels
        self.doc_ids = doc_ids
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        self.num_negatives = num_negatives
        self.max_positives = max_positives

        self.valid_queries = [q for q in queries if
                              q in qrels and any(doc_id in self.doc_id_to_idx for doc_id in qrels[q])]

    def __len__(self) -> int:
        """
        Returns the length of the dataset, which is the number of valid queries with at least one relevant document.
        """
        return len(self.valid_queries)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Retrieves the query embedding, positive and negative document embeddings for a given index.

        Args:
            idx (int): The index of the query to retrieve.

        Returns:
            tuple: A tuple containing:
                - query_embedding (torch.Tensor): The embedding of the query.
                - pos_doc_embeddings (torch.Tensor): The embeddings of relevant (positive) documents.
                - neg_doc_embeddings (torch.Tensor): The embeddings of non-relevant (negative) documents.
                - num_positives (int): The number of positive documents.
        """
        query_id = self.valid_queries[idx]

        query_embedding = self.query_embeddings[self.queries.index(query_id)]
        query_embedding_bert = self.query_embeddings_bert[self.queries.index(query_id)]
        query_embedding_tfidf = self.query_embeddings_tfidf[self.queries.index(query_id)]

        relevant_doc_ids = self.qrels[query_id]
        relevant_doc_indices = [self.doc_id_to_idx[doc_id] for doc_id in relevant_doc_ids
                                if doc_id in self.doc_id_to_idx]

        if len(relevant_doc_indices) > self.max_positives:
            relevant_doc_indices = relevant_doc_indices[:self.max_positives]

        pos_doc_embeddings = self.document_embeddings[relevant_doc_indices]

        neg_doc_indices = []
        while len(neg_doc_indices) < self.num_negatives:
            neg_idx = torch.randint(0, len(self.doc_ids), (1,)).item()
            if neg_idx not in relevant_doc_indices and neg_idx not in neg_doc_indices:
                neg_doc_indices.append(neg_idx)

        neg_doc_embeddings = self.document_embeddings[neg_doc_indices]

        return query_embedding, query_embedding_bert, query_embedding_tfidf, pos_doc_embeddings, neg_doc_embeddings, \
               len(relevant_doc_indices)


def custom_collate_fn(batch: tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function to handle padding and mask creation for positive document embeddings in a batch for the
    Query Adaptive Model.

    Args:
        batch: A list of tuples, where each tuple contains:

        Returns:
            tuple: A tuple containing:
                - queries (torch.Tensor): Tensor of query embeddings by the CDE method of the batch.
                - pos_docs_padded (torch.Tensor): Tensor of padded positive document embeddings of the batch.
                - neg_docs (torch.Tensor): Tensor of negative document embeddings of the batch.
                - pos_masks (torch.Tensor): Tensor of masks indicating the valid positive documents in each padded set.
        """
    queries = torch.stack([item[0] for item in batch])
    max_pos_docs = 5

    pos_docs_padded = []
    pos_masks = []

    for item in batch:
        num_pos = item[3]
        pos_emb = item[1]

        padding_size = max_pos_docs - num_pos
        if padding_size > 0:
            padding = torch.zeros((padding_size, pos_emb.size(1)), device=pos_emb.device)
            pos_emb_padded = torch.cat([pos_emb, padding], dim=0)
        else:
            pos_emb_padded = pos_emb

        pos_docs_padded.append(pos_emb_padded)

        mask = torch.zeros(max_pos_docs, device=pos_emb.device)
        mask[:num_pos] = 1
        pos_masks.append(mask)

    pos_docs_padded = torch.stack(pos_docs_padded)
    pos_masks = torch.stack(pos_masks)

    neg_docs = torch.stack([item[2] for item in batch])

    return queries, pos_docs_padded, neg_docs, pos_masks


def multi_custom_collate_fn(batch: tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                          torch.Tensor, torch.Tensor]:
    """
    Custom collate function to handle padding and mask creation for positive document embeddings in a batch for the
    Query Adaptive Model.

    Args:
        batch: A list of tuples, where each tuple contains:

        Returns:
            A tuple containing:
                - queries (torch.Tensor): Tensor of query embeddings by the CDE method of the batch.
                - pos_docs_padded (torch.Tensor): Tensor of padded positive document embeddings of the batch.
                - neg_docs (torch.Tensor): Tensor of negative document embeddings of the batch.
                - pos_masks (torch.Tensor): Tensor of masks indicating the valid positive documents in each padded set.
        """
    queries = torch.stack([item[0] for item in batch])
    queries_bert = torch.stack([item[1] for item in batch])
    queries_tfidf = torch.stack([item[2] for item in batch])

    max_pos_docs = 5

    pos_docs_padded = []
    pos_masks = []

    for item in batch:
        num_pos = item[5]
        pos_emb = item[3]

        padding_size = max_pos_docs - num_pos
        if padding_size > 0:
            padding = torch.zeros((padding_size, pos_emb.size(1)), device=pos_emb.device)
            pos_emb_padded = torch.cat([pos_emb, padding], dim=0)
        else:
            pos_emb_padded = pos_emb

        pos_docs_padded.append(pos_emb_padded)

        mask = torch.zeros(max_pos_docs, device=pos_emb.device)
        mask[:num_pos] = 1
        pos_masks.append(mask)

    pos_docs_padded = torch.stack(pos_docs_padded)
    pos_masks = torch.stack(pos_masks)

    neg_docs = torch.stack([item[4] for item in batch])

    return queries, queries_bert, queries_tfidf, pos_docs_padded, neg_docs, pos_masks


class MultiPositiveLoss(nn.Module):
    """
    Implements a custom loss that combines a probabilistic loss with focal weighting and a margin penalty to prioritize
    hard positive examples and encourage better separation between positive and negative scores.
    """

    def __init__(self, temperature: float=0.1, margin: float=0.5, alpha: float=2.0) -> None:
        """
        Initializes the MultiPositiveLoss class.

        Args:
            temperature: Scaling factor for the logits to control the sharpness of the softmax distribution (default: 0.1).
            margin: Margin to encourage positive scores to be higher than a set threshold (default: 0.5).
            alpha: Focal-like exponent to give more weight to harder positive examples (default: 2.0).
        """
        super(MultiPositiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.alpha = alpha

    def forward(self, query_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor, pos_mask: torch.Tensor) \
            -> torch.Tensor:
        """
        Computes the loss for query, positive, and negative embeddings using focal loss and margin-based penalty.

        Args:
            query_emb: Embeddings of the queries by CDE method of shape (batch_size, embedding_dim).
            pos_emb: Embeddings of the positive (relevant) documents, shape (batch_size, num_pos, embedding_dim).
            neg_emb: Embeddings of the negative (non-relevant) documents, shape (batch_size, num_neg, embedding_dim).
            pos_mask: Mask for valid positive document embeddings, shape (batch_size, num_pos).

        Returns:
            The computed loss value.
        """
        query_norm = F.normalize(query_emb, dim=-1)
        pos_norm = F.normalize(pos_emb, dim=-1)
        neg_norm = F.normalize(neg_emb, dim=-1)

        pos_scores = torch.sum(query_norm.unsqueeze(1) * pos_norm, dim=-1)  # [batch_size, num_pos]
        neg_scores = torch.sum(query_norm.unsqueeze(1) * neg_norm, dim=-1)  # [batch_size, num_neg]

        pos_scores = pos_scores / self.temperature
        neg_scores = neg_scores / self.temperature

        pos_scores = pos_scores * pos_mask

        neg_max = torch.max(neg_scores, dim=1, keepdim=True)[0]
        exp_scores = torch.cat([pos_scores, neg_scores], dim=1)
        exp_scores = torch.exp(exp_scores - neg_max)

        pos_exp_sum = torch.sum(exp_scores[:, :pos_scores.size(1)] * pos_mask, dim=1)
        neg_exp_sum = torch.sum(exp_scores[:, pos_scores.size(1):], dim=1)

        epsilon = 1e-10
        pos_prob = (pos_exp_sum + epsilon) / (pos_exp_sum + neg_exp_sum + epsilon)

        pos_weight = (1 - pos_prob) ** self.alpha

        margin_loss = F.relu(self.margin - pos_scores) * pos_mask
        margin_loss = margin_loss.mean()

        loss = -torch.log(pos_prob) * pos_weight
        return loss.mean() + margin_loss


def train_query_adaptive_model(model: QueryAdaptiveCDE, dataloader: torch.utils.data.DataLoader, criterion: torch.nn.Module,
                               optimizer: torch.optim.Optimizer, num_epochs: int=30) -> QueryAdaptiveCDE:
    """
    Trains the Query Adaptive CDE model using the provided dataloader, loss function, and optimizer.

    Args:
        model: The query adaptive model to be trained.
        dataloader: DataLoader providing batches of query, positive, and negative document embeddings.
        criterion: The loss function used to compute the training loss.
        optimizer: The optimizer used for updating model weights.
        num_epochs: The number of training epochs.

    Returns:
        The trained model.
    """
    model = model.to(device)
    model.train()

    for _ in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0

        for query_embedding, pos_doc_embeddings, neg_doc_embeddings, pos_mask in tqdm(dataloader):
            query_embedding = query_embedding.to(device)
            pos_doc_embeddings = pos_doc_embeddings.to(device)
            neg_doc_embeddings = neg_doc_embeddings.to(device)
            pos_mask = pos_mask.to(device)

            batch_size, num_pos, emb_dim = pos_doc_embeddings.shape
            pos_doc_embeddings_flat = pos_doc_embeddings.view(-1, emb_dim)
            query_embedding_repeated = query_embedding.repeat_interleave(num_pos, dim=0)
            pos_adaptive_embeddings = model(pos_doc_embeddings_flat, query_embedding_repeated)
            pos_adaptive_embeddings = pos_adaptive_embeddings.view(batch_size, num_pos, emb_dim)

            batch_size, num_negs, emb_dim = neg_doc_embeddings.shape
            neg_doc_embeddings_flat = neg_doc_embeddings.view(-1, emb_dim)
            query_embedding_repeated = query_embedding.repeat_interleave(num_negs, dim=0)
            neg_adaptive_embeddings = model(neg_doc_embeddings_flat, query_embedding_repeated)
            neg_adaptive_embeddings = neg_adaptive_embeddings.view(batch_size, num_negs, emb_dim)

            loss = criterion(query_embedding, pos_adaptive_embeddings, neg_adaptive_embeddings, pos_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        print(f"Average Loss: {epoch_loss / batch_count:.4f}")

    return model


def train_multi_embeddings_query_adaptive_model(model: MultiEmbeddingsQueryAdaptiveCDE, dataloader: torch.utils.data.DataLoader,
                                                criterion: torch.nn.Module, optimizer: torch.optim.Optimizer,
                                                num_epochs: int=30) -> None:
    """
    Trains the Multi-Embeddings-Query Adaptive CDE model using the provided dataloader, loss function, and optimizer.

    Args:
        model: The multi-embeddings-query adaptive model to be trained.
        dataloader: DataLoader providing batches of query, positive, and negative document embeddings.
        criterion: The loss function used to compute the training loss.
        optimizer: The optimizer used for updating model weights.
        num_epochs: The number of training epochs.

    Returns:
        The trained model.
    """
    model = model.to(device)
    model.train()

    for _ in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0

        for query_embedding, query_embedding_bert, query_embedding_tfidf, pos_doc_embeddings, neg_doc_embeddings, \
            pos_mask in tqdm(dataloader):

            query_embedding = query_embedding.to(device)
            query_embedding_bert = query_embedding_bert.to(device)
            query_embedding_tfidf = query_embedding_tfidf.to(device)
            pos_doc_embeddings = pos_doc_embeddings.to(device)
            neg_doc_embeddings = neg_doc_embeddings.to(device)
            pos_mask = pos_mask.to(device)

            batch_size, num_pos, emb_dim = pos_doc_embeddings.shape
            pos_doc_embeddings_flat = pos_doc_embeddings.view(-1, emb_dim)

            query_embedding_repeated = query_embedding.repeat_interleave(num_pos, dim=0)
            query_embedding_bert_repeated = query_embedding_bert.repeat_interleave(num_pos, dim=0)
            query_embedding_tfidf_repeated = query_embedding_tfidf.repeat_interleave(num_pos, dim=0)

            pos_adaptive_embeddings = model(pos_doc_embeddings_flat, query_embedding_repeated,
                                            query_embedding_bert_repeated, query_embedding_tfidf_repeated)
            pos_adaptive_embeddings = pos_adaptive_embeddings.view(batch_size, num_pos, emb_dim)

            batch_size, num_negs, emb_dim = neg_doc_embeddings.shape
            neg_doc_embeddings_flat = neg_doc_embeddings.view(-1, emb_dim)

            query_embedding_repeated = query_embedding.repeat_interleave(num_negs, dim=0)
            query_embedding_bert_repeated = query_embedding_bert.repeat_interleave(num_negs, dim=0)
            query_embedding_tfidf_repeated = query_embedding_tfidf.repeat_interleave(num_negs, dim=0)

            neg_adaptive_embeddings = model(neg_doc_embeddings_flat, query_embedding_repeated,
                                            query_embedding_bert_repeated, query_embedding_tfidf_repeated)
            neg_adaptive_embeddings = neg_adaptive_embeddings.view(batch_size, num_negs, emb_dim)

            loss = criterion(query_embedding, pos_adaptive_embeddings, neg_adaptive_embeddings, pos_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        print(f"Average Loss: {epoch_loss / batch_count:.4f}")

    return model
