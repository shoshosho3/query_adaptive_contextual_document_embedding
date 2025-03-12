import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert self.head_dim * num_heads == embedding_dim, "embedding_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim)
        self.out_linear = nn.Linear(embedding_dim, embedding_dim)

        self.scaling = float(self.head_dim) ** -0.5

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Linear transformations and reshape for multi-head attention
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and apply output linear transformation
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        return self.out_linear(attn_output)


class QueryAdaptiveLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_heads=4):
        super(QueryAdaptiveLayer, self).__init__()
        self.embedding_dim = embedding_dim

        # Multi-head attention layer
        self.attention = MultiHeadAttention(embedding_dim, num_heads)

        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        # Feed-forward network for transformation
        self.ff_network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, document_embeddings, query_embeddings):
        # Reshape inputs for attention
        doc_emb = document_embeddings.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        query_emb = query_embeddings.unsqueeze(1)  # [batch_size, 1, embedding_dim]

        # Apply attention mechanism
        attn_output = self.attention(doc_emb, query_emb, query_emb)

        # First residual connection and layer norm
        doc_emb = self.norm1(doc_emb + attn_output)

        # Feed-forward network
        ff_output = self.ff_network(doc_emb)

        # Second residual connection and layer norm
        output = self.norm2(doc_emb + ff_output)

        return output.squeeze(1)


class QueryAdaptiveCDE(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_heads=4):
        super(QueryAdaptiveCDE, self).__init__()
        self.query_adaptive_layer = QueryAdaptiveLayer(embedding_dim, hidden_dim, num_heads)

    def forward(self, document_embeddings, query_embeddings):
        # Apply query-adaptive transformation with attention
        adaptive_embeddings = self.query_adaptive_layer(document_embeddings, query_embeddings)
        return adaptive_embeddings


class QueryDataset(Dataset):
    def __init__(self, document_embeddings, query_embeddings, queries, qrels, doc_ids, num_negatives=500):
        self.document_embeddings = document_embeddings.to(device)
        self.query_embeddings = query_embeddings.to(device)
        self.queries = queries
        self.qrels = qrels
        self.doc_ids = doc_ids
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query_id = self.queries[idx]
        query_embedding = self.query_embeddings[idx]

        relevant_doc_ids = self.qrels.get(query_id, set())
        relevant_doc_indices = [self.doc_id_to_idx[doc_id] for doc_id in relevant_doc_ids if
                                doc_id in self.doc_id_to_idx]


        if relevant_doc_indices:
            pos_doc_idx = np.random.choice(relevant_doc_indices)
            pos_doc_embedding = self.document_embeddings[pos_doc_idx]
        else:
            pos_doc_embedding = torch.zeros_like(self.document_embeddings[0])

        # Sample multiple negative documents
        neg_doc_indices = []
        while len(neg_doc_indices) < self.num_negatives:
            neg_idx = torch.randint(0, len(self.doc_ids), (1,)).item()
            if neg_idx not in relevant_doc_indices and neg_idx not in neg_doc_indices:
                neg_doc_indices.append(neg_idx)

        neg_doc_embeddings = self.document_embeddings[neg_doc_indices]

        return query_embedding, pos_doc_embedding, neg_doc_embeddings


def train_query_adaptive_model(model, dataloader, optimizer, criterion, num_epochs=30):
    model = model.to(device)
    model.train()
    print(f"Using device: {device}")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        all_scores = []
        all_labels = []

        for query_embedding, pos_doc_embedding, neg_doc_embeddings in tqdm(dataloader):
            # Move to device
            query_embedding = query_embedding.to(device)
            pos_doc_embedding = pos_doc_embedding.to(device)
            neg_doc_embeddings = neg_doc_embeddings.to(device)

            # Forward pass
            pos_adaptive_embedding = model(pos_doc_embedding, query_embedding)

            # Handle negative embeddings as a batch
            batch_size, num_negs, emb_dim = neg_doc_embeddings.shape
            neg_doc_embeddings = neg_doc_embeddings.view(-1, emb_dim)  # Flatten for batch processing
            neg_adaptive_embeddings = model(neg_doc_embeddings, query_embedding.repeat_interleave(num_negs, dim=0))
            neg_adaptive_embeddings = neg_adaptive_embeddings.view(batch_size, num_negs, emb_dim)  # Restore shape

            # Compute loss
            loss = criterion(query_embedding, pos_adaptive_embedding, neg_adaptive_embeddings)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}")
