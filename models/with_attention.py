import torch
import torch.nn as nn
import torch.nn.functional as F


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

        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        return self.out_linear(attn_output)


class QueryAdaptiveLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_heads=4):
        super(QueryAdaptiveLayer, self).__init__()
        self.embedding_dim = embedding_dim

        self.attention = MultiHeadAttention(embedding_dim, num_heads)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.ff_network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim))

    def forward(self, document_embeddings, query_embeddings):
        doc_emb = document_embeddings.unsqueeze(1)
        query_emb = query_embeddings.unsqueeze(1)

        attn_output = self.attention(doc_emb, query_emb, query_emb)
        doc_emb = self.norm1(doc_emb + attn_output)
        ff_output = self.ff_network(doc_emb)

        output = self.norm2(doc_emb + ff_output)

        return output.squeeze(1)


class QueryAdaptiveCDE(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_heads=4):
        super(QueryAdaptiveCDE, self).__init__()
        self.query_adaptive_layer = QueryAdaptiveLayer(embedding_dim, hidden_dim, num_heads)

    def forward(self, document_embeddings, query_embeddings):
        adaptive_embeddings = self.query_adaptive_layer(document_embeddings, query_embeddings)
        return adaptive_embeddings


class MultiEmbeddingsQueryAdaptiveLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, query_embeddings_tensor_tfidf, num_heads=4):
        super(MultiEmbeddingsQueryAdaptiveLayer, self).__init__()
        self.embedding_dim = embedding_dim

        self.first_layer = nn.Sequential(
            nn.Linear(query_embeddings_tensor_tfidf.size(1), self.embedding_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2))

        self.second_layer = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2))

        self.third_layer = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2))

        self.ff_network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim))

        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def compute_combined_query_embedding(self, query_embeddings, query_embeddings_bert, query_embeddings_tfidf):
        query_emb = query_embeddings.unsqueeze(1).float()
        query_emb_bert = query_embeddings_bert.unsqueeze(1).float()
        query_emb_tfidf = query_embeddings_tfidf.unsqueeze(1).float()

        query_emb_tfidf = self.first_layer(query_emb_tfidf)
        query_emb_tfidf_bert = torch.cat((query_emb_tfidf, query_emb_bert), dim=-1)
        query_emb_tfidf_bert = self.second_layer(query_emb_tfidf_bert)
        query_fully_combined_emb = torch.concat((query_emb_tfidf_bert, query_emb), dim=-1)
        query_fully_combined_emb = self.third_layer(query_fully_combined_emb)

        return query_fully_combined_emb

    def forward(self, document_embeddings, query_embeddings, query_embeddings_bert, query_embeddings_tfidf):
        doc_emb = document_embeddings.unsqueeze(1)

        query_fully_combined_emb = self.compute_combined_query_embedding(query_embeddings, query_embeddings_bert,
                                                                         query_embeddings_tfidf)

        attention_output = self.attention(doc_emb, query_fully_combined_emb, query_fully_combined_emb)
        doc_emb = self.norm1(doc_emb + attention_output)
        ff_output = self.ff_network(doc_emb)

        output = self.norm2(doc_emb + ff_output)

        return output.squeeze(1)


class MultiEmbeddingsQueryAdaptiveCDE(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, query_embeddings_tfidf, num_heads=4):
        super(MultiEmbeddingsQueryAdaptiveCDE, self).__init__()
        self.multi_embeddings_query_adaptive_layer = MultiEmbeddingsQueryAdaptiveLayer(embedding_dim, hidden_dim,
                                                                                       query_embeddings_tfidf, num_heads)

    def forward(self, document_embeddings, query_embeddings, query_embeddings_bert, query_embeddings_tfidf):
        adaptive_embeddings = self.multi_embeddings_query_adaptive_layer(document_embeddings, query_embeddings,
                                                                         query_embeddings_bert, query_embeddings_tfidf)
        return adaptive_embeddings




