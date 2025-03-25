import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadAttention(nn.Module):
    """
       Implements the multi-head attention mechanism as described in the Transformer architecture.
       The attention mechanism splits the embeddings into multiple heads, computes attention for each head,
       and then concatenates the results to form the output.

       This class allows the model to jointly attend to information from different representation subspaces at different positions.
       """
    def __init__(self, embedding_dim: int, num_heads: int):
        """
        Initializes the MultiHeadAttention layer with the given embedding dimension and number of attention heads.

        Args:
            embedding_dim: The dimension of the input embeddings.
            num_heads: The number of attention heads. Each head attends to a different part of the input.
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert self.head_dim * num_heads == embedding_dim, "embedding_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim)
        self.out_linear = nn.Linear(embedding_dim, embedding_dim)

        self.scaling = float(self.head_dim) ** -0.5

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the multi-head attention mechanism. It computes attention scores, applies them to
        the values, and concatenates the results across all heads.

        Args:
            query: The query tensor of shape (batch_size, seq_length, embedding_dim).
            key: The key tensor of shape (batch_size, seq_length, embedding_dim).
            value: The value tensor of shape (batch_size, seq_length, embedding_dim).

        Returns:
            The output tensor after applying multi-head attention, of shape (batch_size, seq_length, embedding_dim).
        """
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
    """
    Implements a query-adaptive transformation layer using multi-head attention. This layer adjusts document embeddings
    based on query embeddings, allowing the model to learn representations that are influenced by the queries.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, num_heads: int=4):
        """
        Initializes the QueryAdaptiveLayer with attention and feed-forward networks.

        Args:
            embedding_dim: The dimension of the input embeddings (for both queries and documents).
            hidden_dim: The dimension of the hidden layer in the feed-forward network.
            num_heads: The number of attention heads (default: 4).
        """
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

    def forward(self, document_embeddings: torch.Tensor, query_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass that adjusts document embeddings based on query embeddings using multi-head attention
        and a feed-forward network.

        Args:
            document_embeddings: Tensor containing document embeddings, shape (batch_size, embedding_dim).
            query_embeddings: Tensor containing query embeddings, shape (batch_size, embedding_dim).

        Returns:
            The adjusted document embeddings of shape (batch_size, embedding_dim), adapted based on query embeddings.
        """
        doc_emb = document_embeddings.unsqueeze(1)
        query_emb = query_embeddings.unsqueeze(1)

        attn_output = self.attention.forward(doc_emb, query_emb, query_emb)
        doc_emb = self.norm1(doc_emb + attn_output)
        ff_output = self.ff_network(doc_emb)

        output = self.norm2(doc_emb + ff_output)

        return output.squeeze(1)


class QueryAdaptiveCDE(nn.Module):
    """
    Implements a Query-Adaptive Contextual Document Embedding (QACDE) model that adjusts document embeddings based on
    query embeddings using the QueryAdaptiveLayer. This allows the model to create document embeddings that are
    influenced by the document, its context and the query.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, num_heads: int=4):
        """
        Initializes the QueryAdaptiveCDE model with a QueryAdaptiveLayer for query-based document embedding adaptation.

        Args:
            embedding_dim: The dimension of the input embeddings (for both documents and queries).
            hidden_dim: The dimension of the hidden layer in the feed-forward network.
            num_heads: The number of attention heads for multi-head attention.
        """
        super(QueryAdaptiveCDE, self).__init__()
        self.query_adaptive_layer = QueryAdaptiveLayer(embedding_dim, hidden_dim, num_heads)

    def forward(self, document_embeddings: torch.Tensor, query_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Adjusts document embeddings based on query embeddings using the QueryAdaptiveLayer.

        Args:
            document_embeddings (torch.Tensor): Tensor of document embeddings, shape (batch_size, embedding_dim).
            query_embeddings (torch.Tensor): Tensor of query embeddings, shape (batch_size, embedding_dim).

        Returns:
            The adjusted document embeddings of shape (batch_size, embedding_dim), influenced by the query.
        """
        adaptive_embeddings = self.query_adaptive_layer.forward(document_embeddings, query_embeddings)
        return adaptive_embeddings


class MultiEmbeddingsQueryAdaptiveLayer(nn.Module):
    """
    Implements a query-adaptive transformation layer using multi-head attention. This layer adjusts document embeddings
    based on different query embeddings, allowing the model to learn representations that are influenced by the queries
    and to leverage the strengths of the different query embeddings.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, query_embeddings_tensor_tfidf: torch.Tensor, num_heads: int=4):
        """
        Initializes the MultiEmbeddingsQueryAdaptiveLayer with attention and feed-forward networks.

        Args:
            embedding_dim: The dimension of the input embeddings (for both queries and documents).
            hidden_dim: The dimension of the hidden layer in the feed-forward network.
            query_embeddings_tensor_tfidf: Tensor of the query embeddings by TF-IDF method of shape (queries_num, embedding_dim).
            num_heads: The number of attention heads.
        """
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

    def compute_combined_query_embedding(self, query_embeddings: torch.Tensor, query_embeddings_bert: torch.Tensor,
                                         query_embeddings_tfidf: torch.Tensor) -> torch.Tensor:
        """
        Combines multiple query embeddings (CDE, BERT, and TF-IDF) into a single unified representation. This method
        processes the TF-IDF embedding, concatenates it with the BERT embedding, and further combines it with the
        original query embedding. The output is a query embedding that leverages the strengths of the different query
        representations.

        Args:
            query_embeddings: The original query embeddings of shape (batch_size, embedding_dim).
            query_embeddings_bert: The query embeddings generated by a BERT model of shape (batch_size, embedding_dim).
            query_embeddings_tfidf: The query embeddings generated by the TF-IDF method of shape (batch_size, tfidf_dim).

        Returns:
            The combined query embedding of shape (batch_size, embedding_dim), leveraging the BERT, TF-IDF, and original
             query embeddings.
        """
        query_emb = query_embeddings.unsqueeze(1).float()
        query_emb_bert = query_embeddings_bert.unsqueeze(1).float()
        query_emb_tfidf = query_embeddings_tfidf.unsqueeze(1).float()

        query_emb_tfidf = self.first_layer(query_emb_tfidf)
        query_emb_tfidf_bert = torch.cat((query_emb_tfidf, query_emb_bert), dim=-1)
        query_emb_tfidf_bert = self.second_layer(query_emb_tfidf_bert)
        query_fully_combined_emb = torch.concat((query_emb_tfidf_bert, query_emb), dim=-1)
        query_fully_combined_emb = self.third_layer(query_fully_combined_emb)

        return query_fully_combined_emb

    def forward(self, document_embeddings: torch.Tensor, query_embeddings: torch.Tensor,
                query_embeddings_bert: torch.Tensor, query_embeddings_tfidf: torch.Tensor) -> torch.Tensor:
        """
        Adjusts document embeddings based on query embeddings using the QueryAdaptiveLayer.

        Args:
            document_embeddings: Tensor of document embeddings by CDE method of shape (batch_size, embedding_dim).
            query_embeddings: Tensor of query embeddings by CDE method of shape (batch_size, embedding_dim).
            query_embeddings_bert: Tensor of query embeddings by BERT method of shape (batch_size, embedding_dim).
            query_embeddings_tfidf: Tensor of query embeddings by TF-IDF method of shape (batch_size, embedding_dim)

        Returns:
            The adjusted document embeddings of shape (batch_size, embedding_dim), influenced by the query.
        """
        doc_emb = document_embeddings.unsqueeze(1)

        query_fully_combined_emb = self.compute_combined_query_embedding(query_embeddings, query_embeddings_bert,
                                                                         query_embeddings_tfidf)

        attention_output = self.attention.forward(doc_emb, query_fully_combined_emb, query_fully_combined_emb)
        doc_emb = self.norm1(doc_emb + attention_output)
        ff_output = self.ff_network(doc_emb)

        output = self.norm2(doc_emb + ff_output)

        return output.squeeze(1)


class MultiEmbeddingsQueryAdaptiveCDE(nn.Module):
    """
    Implements a Multi-Embeddings-Query Adaptive Contextual Document Embedding (MEQACDE) model that adjusts document
    embeddings based on query embeddings using the MultiEmbeddingsQueryAdaptiveLayer. This allows the model to create
    document embeddings that are influenced by the document, its context and the several representations of the query.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, query_embeddings_tfidf: torch.Tensor, num_heads: int=4):
        """
        Initializes the QueryAdaptiveCDE model with a QueryAdaptiveLayer for query-based document embedding adaptation.

        Args:
            embedding_dim: The dimension of the input embeddings (for both documents and queries).
            hidden_dim: The dimension of the hidden layer in the feed-forward network.
            query_embeddings_tfidf: Tensor of the query embeddings by the TF-IDF method of shape (num_queries, embedding_dim).
            num_heads: The number of attention heads for multi-head attention.
        """
        super(MultiEmbeddingsQueryAdaptiveCDE, self).__init__()
        self.multi_embeddings_query_adaptive_layer = MultiEmbeddingsQueryAdaptiveLayer(embedding_dim, hidden_dim,
                                                                                       query_embeddings_tfidf, num_heads)

    def forward(self, document_embeddings: torch.Tensor, query_embeddings: torch.Tensor,
                query_embeddings_bert: torch.Tensor, query_embeddings_tfidf: torch.Tensor) -> torch.Tensor:
        """
        Adjusts document embeddings based on different query embeddings using the MultiEmbeddingsQueryAdaptiveLayer.

        Args:
            document_embeddings: Tensor of document embeddings by the CDE method of shape (batch_size, embedding_dim).
            query_embeddings: Tensor of query embeddings by the CDE method of shape (batch_size, embedding_dim).
            query_embeddings_bert: Tensor of query embeddings by the BERT method of shape (batch_size, embedding_dim).
            query_embeddings_tfidf: Tensor of query embeddings by the TF-IDF method of shape (batch_size, embedding_dim).

        Returns:
            The adjusted document embeddings of shape (batch_size, embedding_dim), influenced by  different
            representations of the query.
        """
        adaptive_embeddings = self.multi_embeddings_query_adaptive_layer.forward(document_embeddings, query_embeddings,
                                                                         query_embeddings_bert, query_embeddings_tfidf)
        return adaptive_embeddings




