import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Previous MultiHeadAttention, QueryAdaptiveLayer, and QueryAdaptiveCDE classes remain unchanged

class QueryDataset(Dataset):
    def __init__(self, document_embeddings, query_embeddings, queries, qrels, doc_ids, num_negatives=500,
                 max_positives=50):
        self.document_embeddings = document_embeddings.to(device)
        self.query_embeddings = query_embeddings.to(device)
        self.queries = queries
        self.qrels = qrels
        self.doc_ids = doc_ids
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        self.num_negatives = num_negatives
        self.max_positives = max_positives

        # Filter out queries with no relevant documents
        self.valid_queries = [q for q in queries if
                              q in qrels and any(doc_id in self.doc_id_to_idx for doc_id in qrels[q])]

    def __len__(self):
        return len(self.valid_queries)

    def __getitem__(self, idx):
        query_id = self.valid_queries[idx]
        query_embedding = self.query_embeddings[self.queries.index(query_id)]

        # Get all relevant document indices for this query
        relevant_doc_ids = self.qrels[query_id]
        relevant_doc_indices = [self.doc_id_to_idx[doc_id] for doc_id in relevant_doc_ids
                                if doc_id in self.doc_id_to_idx]

        # Limit the number of positive documents if necessary
        if len(relevant_doc_indices) > self.max_positives:
            relevant_doc_indices = relevant_doc_indices[:self.max_positives]

        # Get embeddings for all positive documents
        pos_doc_embeddings = self.document_embeddings[relevant_doc_indices]

        # Sample negative documents
        neg_doc_indices = []
        while len(neg_doc_indices) < self.num_negatives:
            neg_idx = torch.randint(0, len(self.doc_ids), (1,)).item()
            if neg_idx not in relevant_doc_indices and neg_idx not in neg_doc_indices:
                neg_doc_indices.append(neg_idx)

        neg_doc_embeddings = self.document_embeddings[neg_doc_indices]

        return query_embedding, pos_doc_embeddings, neg_doc_embeddings, len(relevant_doc_indices)


def custom_collate_fn(batch):
    queries = torch.stack([item[0] for item in batch])

    # Get the maximum number of positive documents in this batch
    # max_pos_docs = max(item[3] for item in batch)
    max_pos_docs = 5

    # Pad positive documents
    pos_docs_padded = []
    pos_masks = []
    for item in batch:
        num_pos = item[3]
        pos_emb = item[1]

        # Create padding
        padding_size = max_pos_docs - num_pos
        if padding_size > 0:
            padding = torch.zeros((padding_size, pos_emb.size(1)), device=pos_emb.device)
            pos_emb_padded = torch.cat([pos_emb, padding], dim=0)
        else:
            pos_emb_padded = pos_emb

        pos_docs_padded.append(pos_emb_padded)

        # Create mask (1 for real documents, 0 for padding)
        mask = torch.zeros(max_pos_docs, device=pos_emb.device)
        mask[:num_pos] = 1
        pos_masks.append(mask)

    pos_docs_padded = torch.stack(pos_docs_padded)
    pos_masks = torch.stack(pos_masks)
    neg_docs = torch.stack([item[2] for item in batch])

    return queries, pos_docs_padded, neg_docs, pos_masks

class MultiPositiveLoss(nn.Module):
    def __init__(self, temperature=0.1, margin=0.5, alpha=2.0):
        super(MultiPositiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.alpha = alpha  # Focal-like exponent

    def forward(self, query_emb, pos_emb, neg_emb, pos_mask):
        query_norm = F.normalize(query_emb, dim=-1)
        pos_norm = F.normalize(pos_emb, dim=-1)
        neg_norm = F.normalize(neg_emb, dim=-1)

        # Compute cosine similarities
        pos_scores = torch.sum(query_norm.unsqueeze(1) * pos_norm, dim=-1)  # [batch_size, num_pos]
        neg_scores = torch.sum(query_norm.unsqueeze(1) * neg_norm, dim=-1)  # [batch_size, num_neg]

        # Apply temperature scaling
        pos_scores = pos_scores / self.temperature
        neg_scores = neg_scores / self.temperature

        # Mask out padding in positive documents
        pos_scores = pos_scores * pos_mask

        # Compute max for numerical stability
        neg_max = torch.max(neg_scores, dim=1, keepdim=True)[0]
        exp_scores = torch.cat([pos_scores, neg_scores], dim=1)
        exp_scores = torch.exp(exp_scores - neg_max)

        # Handle masked positives
        pos_exp_sum = torch.sum(exp_scores[:, :pos_scores.size(1)] * pos_mask, dim=1)
        neg_exp_sum = torch.sum(exp_scores[:, pos_scores.size(1):], dim=1)

        # Compute probabilities
        epsilon = 1e-10
        pos_prob = (pos_exp_sum + epsilon) / (pos_exp_sum + neg_exp_sum + epsilon)

        # **Focal-like weighting:** Punish low-ranked positives more
        pos_weight = (1 - pos_prob) ** self.alpha

        # **Margin loss:** Encourage positive scores to be higher than a margin
        margin_loss = F.relu(self.margin - pos_scores) * pos_mask
        margin_loss = margin_loss.mean()

        # Final loss: log-loss with focal weighting + margin penalty
        loss = -torch.log(pos_prob) * pos_weight
        return loss.mean() + margin_loss


def train_query_adaptive_model(model, dataloader, optimizer, num_epochs=30, eval_every_n_batches=1):
    model = model.to(device)
    model.train()
    criterion = MultiPositiveLoss()
    print(f"Using device: {device}")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0

        total_map = 0

        for query_embedding, pos_doc_embeddings, neg_doc_embeddings, pos_mask in tqdm(dataloader):
            # Move to device
            query_embedding = query_embedding.to(device)
            pos_doc_embeddings = pos_doc_embeddings.to(device)
            neg_doc_embeddings = neg_doc_embeddings.to(device)
            pos_mask = pos_mask.to(device)

            # Forward pass for positive documents
            batch_size, num_pos, emb_dim = pos_doc_embeddings.shape
            pos_doc_embeddings_flat = pos_doc_embeddings.view(-1, emb_dim)
            query_embedding_repeated = query_embedding.repeat_interleave(num_pos, dim=0)
            pos_adaptive_embeddings = model(pos_doc_embeddings_flat, query_embedding_repeated)
            pos_adaptive_embeddings = pos_adaptive_embeddings.view(batch_size, num_pos, emb_dim)

            # Forward pass for negative documents
            batch_size, num_negs, emb_dim = neg_doc_embeddings.shape
            neg_doc_embeddings_flat = neg_doc_embeddings.view(-1, emb_dim)
            query_embedding_repeated = query_embedding.repeat_interleave(num_negs, dim=0)
            neg_adaptive_embeddings = model(neg_doc_embeddings_flat, query_embedding_repeated)
            neg_adaptive_embeddings = neg_adaptive_embeddings.view(batch_size, num_negs, emb_dim)

            # Compute loss
            loss = criterion(query_embedding, pos_adaptive_embeddings, neg_adaptive_embeddings, pos_mask)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            # Calculate MAP periodically on a subset of the data
            # if batch_count % eval_every_n_batches == 0:
            #     with torch.no_grad():
            #         # Sample additional negative documents for MAP calculation
            #         num_extra_negs = 1  # More realistic evaluation while keeping it fast
            #         rand_indices = torch.randint(0, len(dataloader.dataset.document_embeddings),
            #                                      (batch_size, num_extra_negs)).to(device)
            #         extra_neg_docs = dataloader.dataset.document_embeddings[rand_indices]
            #
            #         # Get adaptive embeddings for extra negatives
            #         query_emb_for_extra = query_embedding.repeat_interleave(num_extra_negs, dim=0)
            #         extra_neg_adaptive = model(extra_neg_docs.view(-1, emb_dim), query_emb_for_extra)
            #         extra_neg_adaptive = extra_neg_adaptive.view(batch_size, num_extra_negs, emb_dim)
            #
            #         # Combine all documents for evaluation
            #         all_docs = torch.cat([pos_adaptive_embeddings, neg_adaptive_embeddings, extra_neg_adaptive], dim=1)
            #
            #         # Calculate scores
            #         query_norm = F.normalize(query_embedding, dim=-1)
            #         doc_norms = F.normalize(all_docs, dim=-1)
            #         scores = torch.bmm(doc_norms, query_norm.unsqueeze(-1)).squeeze(-1)
            #
            #         # Create labels
            #         labels = torch.zeros_like(scores)
            #         labels[:, :num_pos] = pos_mask
            #
            #         # Calculate MAP
            #         map_score = 0
            #         for i in range(batch_size):
            #             # valid_mask = torch.cat([pos_mask[i], torch.ones(num_negs + num_extra_negs)]).bool()
            #             valid_mask = torch.cat(
            #                 [pos_mask[i], torch.ones(num_negs + num_extra_negs, device=device)]).bool()
            #             map_score += average_precision_score(
            #                 labels[i][valid_mask].cpu().numpy(),
            #                 scores[i][valid_mask].cpu().numpy()
            #             )
            #         map_score /= batch_size
            #
            #         total_map += map_score

        print(f"Epoch {epoch + 1}, Average Loss: {epoch_loss / batch_count:.4f}")

    return model