from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def process_ex_document(ex: dict) -> dict:
    """
    Formats a document into a string containing its title and text.

    Args:
        ex (dict): A dictionary with document fields, typically 'title' and 'text'.

    Returns:
        str: A combined string of title and text.
    """

    ex["text"] = f"{ex['title']} {ex['text']}"
    return ex


def tokenize(tokenizer, documents: list, prefix: str, device, max_length: int = 512) -> list:
    """
    Tokenizes a list of documents for input into a model.

    Args:
        tokenizer: Tokenizer object to process text.
        documents (list): List of documents as strings.
        max_length (int): Maximum number of tokens per document (default: 512).

    Returns:
        dict: A dictionary of PyTorch tensors with tokenized documents.
        :param max_length:
        :param tokenizer:
        :param documents:
        :param prefix: Prefix to add to each document.
    """
    tokenized_docs = []
    for doc in tqdm(documents, desc="Tokenizing"):
        tokenized_docs.append(tokenizer(
            prefix + doc.cpu(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device))
    return tokenized_docs