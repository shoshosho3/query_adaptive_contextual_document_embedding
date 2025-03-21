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
    def __init__(self, train_queries, test_queries=None, dev_queries=None, embedding_dim=768,
                 bert_model_name='bert-base-uncased'):

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

    def get_bert_embedding(self, query):
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        query_embedding = outputs.last_hidden_state[:, 0, :]

        if self.projection_layer is not None:
            query_embedding = self.projection_layer(query_embedding)

        return query_embedding.squeeze()

    def get_queries_embeddings(self):
        self.train_queries_embeddings = torch.stack([self.get_bert_embedding(query) for query in self.train_queries],
                                                    dim=0)

        if self.test_queries is not None:
            self.test_queries_embeddings = torch.stack([self.get_bert_embedding(query) for query in self.test_queries],
                                                       dim=0)

        if self.dev_queries is not None:
            self.dev_queries_embeddings = torch.stack([self.get_bert_embedding(query) for query in self.dev_queries],
                                                      dim=0)


class QueryTFIDFEmbedder:
    def __init__(self, corpus, train_queries, max_dim=3000, test_queries=None, dev_queries=None):
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

    def get_tfidf_scores(self, query):
        tfidf_vector = self.vectorizer.transform([query.lower()])
        return tfidf_vector.toarray()[0]

    def get_queries_embeddings(self):
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


def find_last_query_index(dataset_name, model_name):
    index = 1
    while True:
        try:
            with open(f'{dataset_name}/query_embeddings_train_{dataset_name}_{model_name}_{index}.pkl', 'rb') as f:
                index += 1
        except FileNotFoundError:
            return index


def save_queries_embeddings_pickle(dataset_name: str, models: list, model_names: list):
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
