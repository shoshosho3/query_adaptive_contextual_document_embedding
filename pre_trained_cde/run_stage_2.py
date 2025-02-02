from pre_trained_cde.stages_utils import *
from tqdm import tqdm
import torch
import os

QUERIES = 1
TRAIN = "train"
DEV = "dev"
TEST = "test"


def embed_documents(tokenized_docs, model, dataset_embeddings, description):
    """Embed the tokenized documents using the second stage model."""
    doc_embeddings_list = []
    with torch.no_grad():
        for tokenized_doc in tqdm(tokenized_docs, desc=description):
            doc_embeddings = model.second_stage_model(
                input_ids=tokenized_doc["input_ids"],
                attention_mask=tokenized_doc["attention_mask"],
                dataset_embeddings=dataset_embeddings,
            )
            doc_embeddings_list.append(doc_embeddings)
    return doc_embeddings_list


def process_and_tokenize_queries(split, tokenizer, dataset_embeddings, stage_model, stage, device):
    """Tokenize and embed query data with progress tracking."""
    queries = split[QUERIES].values()
    tokenized_queries = tokenize(tokenizer, queries, QUERY_PREFIX, device)
    query_embeddings_list = embed_documents(
        tokenized_queries,
        stage_model,
        dataset_embeddings,
        f"Processing embeddings for {stage} queries"
    )
    return query_embeddings_list


def process_corpus(corpus, tokenizer, device):
    """Tokenize the corpus and prepare it for embeddings."""
    corpus_list = list(corpus.items())
    docs = [process_ex_document(doc[1])['text'] for doc in corpus_list]
    tokenized_docs = tokenize(tokenizer, docs, DOCUMENT_PREFIX, device)
    return tokenized_docs


def process_corpus_embeddings(tokenized_docs, model, dataset_embeddings):
    """Embed the tokenized corpus."""
    return embed_documents(tokenized_docs, model, dataset_embeddings, "Processing document embeddings")


def process_split_section(split_method, tokenizer, model, dataset_embeddings, split_type, stage, device):
    """Wrapper for processing train/dev/test splits."""
    if split_method():
        return process_and_tokenize_queries(split_type, tokenizer, dataset_embeddings, model, stage, device)
    return None


def run_stage_2(corpus, model, tokenizer, device, dataset_embeddings, get_split):
    """
    Main function for running stage 2 of the pipeline:
    Process documents, tokenize, and generate embeddings for various splits (train/dev/test).
    """

    # Process corpus and generate document embeddings.
    tokenized_docs = process_corpus(corpus, tokenizer, device)
    doc_embeddings_list = process_corpus_embeddings(tokenized_docs, model, dataset_embeddings)

    # Process train, dev, and test splits
    train_processed = process_and_tokenize_queries(get_split.train, tokenizer, dataset_embeddings, model, TRAIN, device)
    dev_processed = process_split_section(get_split.has_dev, tokenizer, model, dataset_embeddings, get_split.dev, DEV, device)
    test_processed = process_split_section(get_split.has_test, tokenizer, model, dataset_embeddings, get_split.test, TEST, device)

    return doc_embeddings_list, train_processed, dev_processed, test_processed
