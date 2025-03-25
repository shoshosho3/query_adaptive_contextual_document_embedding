import nltk
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from pre_trained_cde.save_pre_trained import check_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from warnings import filterwarnings
from pre_trained_cde.stages_utils import process_ex_document

nltk.download('punkt_tab')
filterwarnings("ignore")


class QueryBertEmbedder:
    """
    A class for generating query embeddings using a BERT model. This class processes queries (train, test, and dev)
    through a pre-trained BERT model and optionally projects the embeddings to a specified dimensionality.
    """
    def __init__(self, train_queries: list, test_queries: list=None, dev_queries: list=None, embedding_dim: int=768,
                 bert_model_name: str='bert-base-uncased'):
        """
        Initializes the QueryBertEmbedder class with a BERT model and prepares for query embeddings generation.

        Args:
            train_queries: A list of training queries to embed.
            test_queries: A list of test queries to embed.
            dev_queries: A list of development/validation queries to embed.
            embedding_dim: The dimension to which BERT embeddings will be projected. If it is different from the BERT
            model's hidden size, a projection layer will be used.
            bert_model_name: The name of the pre-trained BERT model to use.
        """

        self.train_queries = train_queries
        self.test_queries = test_queries
        self.dev_queries = dev_queries

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)

        self.embedding_dim = embedding_dim

        self.projection_layer = nn.Linear(self.bert_model.config.hidden_size, self.embedding_dim) \
            if self.embedding_dim != self.bert_model.config.hidden_size else None

        self.train_queries_embeddings = None
        self.test_queries_embeddings = None
        self.dev_queries_embeddings = None

    def get_bert_embedding(self, query: str) -> torch.Tensor:
        """
        Generates the BERT embedding for a single query, optionally projecting it to the specified embedding dimension.

        Args:
            query: The query string to be embedded using BERT.

        Returns:
            The BERT embedding for the query. If a projection layer is specified, the embedding is projected to the
            specified dimension. The tensor has shape (embedding_dim).
        """
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        query_embedding = outputs.last_hidden_state[:, 0, :]

        if self.projection_layer is not None:
            query_embedding = self.projection_layer(query_embedding)

        return query_embedding.squeeze()

    def get_queries_embeddings(self):
        """
        Generates and stores the embeddings for train, test, and dev queries using BERT. The embeddings are stored as
        tensors for each set of queries (train, test, dev).
        """
        self.train_queries_embeddings = torch.stack([self.get_bert_embedding(query) for query in self.train_queries],
                                                    dim=0)

        if self.test_queries is not None:
            self.test_queries_embeddings = torch.stack([self.get_bert_embedding(query) for query in self.test_queries],
                                                       dim=0)

        if self.dev_queries is not None:
            self.dev_queries_embeddings = torch.stack([self.get_bert_embedding(query) for query in self.dev_queries],
                                                      dim=0)


class QueryTFIDFEmbedder:
    """
    A class for generating query embeddings using TF-IDF vectorization. This class processes queries (train, test, and
    dev) by converting them into TF-IDF feature vectors based on the corpus.
    """
    def __init__(self, corpus: dict, train_queries: list, max_dim: int=3000, test_queries: list=None, dev_queries: list=None):
        """
        Initializes the QueryTFIDFEmbedder class with a TF-IDF vectorizer and prepares for query embeddings generation.

        Args:
            corpus: A dictionary of documents used to fit the TF-IDF vectorizer.
            train_queries: A list of training queries to embed.
            max_dim Maximum number of features for the TF-IDF vectorizer.
            test_queries: A list of test queries to embed.
            dev_queries: A list of development/validation queries to embed.
        """
        self.corpus = corpus
        self.train_queries = train_queries
        self.test_queries = test_queries
        self.dev_queries = dev_queries

        self.vectorizer = TfidfVectorizer(tokenizer=lambda doc: word_tokenize(doc.lower()), lowercase=False,
                                          max_features=max_dim)

        self.vectorizer.fit([(process_ex_document(doc))['text'].lower() for doc in self.corpus])

        self.train_queries_embeddings = None
        self.test_queries_embeddings = None
        self.dev_queries_embeddings = None

    def get_tfidf_scores(self, query: str) -> np.array:
        """
        Generates the TF-IDF feature vector for a single query.

        Args:
            query: The query string to be vectorized using the TF-IDF vectorizer.

        Returns:
            A 1D array representing the TF-IDF scores for the query.
        """
        tfidf_vector = self.vectorizer.transform([query.lower()])
        return tfidf_vector.toarray()[0]

    def get_queries_embeddings(self):
        """
        Generates and stores the embeddings for train, test, and dev queries using TF-IDF vectorization. The embeddings
        are stored as tensors for each set of queries (train, test, dev).
        """
        self.train_queries_embeddings = torch.stack(
            [torch.tensor(self.get_tfidf_scores(query) / np.linalg.norm(self.get_tfidf_scores(query)))
             if np.linalg.norm(self.get_tfidf_scores(query)) != 0 else torch.tensor(self.get_tfidf_scores(query))
             for query in self.train_queries], dim=0)

        if self.test_queries is not None:
            self.test_queries_embeddings = torch.stack(
                [torch.tensor(self.get_tfidf_scores(query) / np.linalg.norm(self.get_tfidf_scores(query)))
                 if np.linalg.norm(self.get_tfidf_scores(query)) != 0 else torch.tensor(self.get_tfidf_scores(query))
                 for query in self.test_queries], dim=0)

        if self.dev_queries is not None:
            self.dev_queries_embeddings = torch.stack(
                [torch.tensor(self.get_tfidf_scores(query) / np.linalg.norm(self.get_tfidf_scores(query)))
                 if np.linalg.norm(self.get_tfidf_scores(query)) != 0 else torch.tensor(self.get_tfidf_scores(query))
                 for query in self.dev_queries], dim=0)


def find_last_query_index(dataset_name: str, model_name: str) -> int:
    """
    Finds the next available index for saving query embeddings in the specified dataset and model. This function
    increments the index until it finds a non-existing query embedding file.

    Args:
        dataset_name: The name of the dataset being used.
        model_name: The name of the model for which query embeddings are saved.

    Returns:
        The next available index for saving query embeddings.
    """
    index = 1
    while True:
        try:
            with open(f'{dataset_name}/query_embeddings_train_{dataset_name}_{model_name}_{index}.pkl', 'rb') as f:
                index += 1
        except FileNotFoundError:
            return index


def save_queries_embeddings_pickle(dataset_name: str, models: list, model_names: list):
    """
    Saves the query embeddings (train, dev, and test) generated by multiple models into pickle files. The function
    automatically increments the index for each model to avoid overwriting existing files.

    Args:
        dataset_name: The name of the dataset where the query embeddings will be saved.
        models: A list of model instances that generate the query embeddings.
        model_names: A list of corresponding names for the models, used to name the output files.
    """
    check_directory(dataset_name)

    for model, name in zip(models, model_names):
        model.get_queries_embeddings()
        index = find_last_query_index(dataset_name, name)

        with open(f'{dataset_name}/query_embeddings_train_{dataset_name}_{name}_{index}.pkl', 'wb') as f:
            pkl.dump(model.train_queries_embeddings, f)

        if model.dev_queries is not None:
            with open(f'{dataset_name}/query_embeddings_dev_{dataset_name}_{name}_{index}.pkl', 'wb') as f:
                pkl.dump(model.dev_queries_embeddings, f)

        if model.test_queries is not None:
            with open(f'{dataset_name}/query_embeddings_test_{dataset_name}_{name}_{index}.pkl', 'wb') as f:
                pkl.dump(model.test_queries_embeddings, f)

        print(f"Queries Were Embedded Successfully by {name} Method!")
