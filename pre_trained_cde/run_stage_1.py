import random
import torch
import transformers
from tqdm import tqdm
from pre_trained_cde.stages_utils import *


def get_minicorpus(corpus, size: int) -> list:
    """
    Extracts a random subset of documents from the corpus.

    Args:
        corpus (dict): The full corpus as a dictionary of documents.
        size (int): The number of documents to sample.

    Returns:
        list: A list of documents formatted as strings using `process_ex_document`.
    """
    # Convert corpus to a list for indexed access, then sample random documents

    corpus_list = list(corpus.items())
    random_indices = random.choices(range(len(corpus_list)), k=size)

    # Use document processing function to generate titles + text
    return [process_ex_document(corpus_list[random_indices[i]][1])['text'] for i in range(len(random_indices))]


def generate_embeddings(model, tokenized_docs: transformers.tokenization_utils_base.BatchEncoding,
                        device, batch_size: int = 32) -> torch.Tensor:
    """
    Computes dataset embeddings by running tokenized documents through the model in batches.

    Args:
        model: The ML model that generates embeddings.
        tokenized_docs (dict): Tokenized input documents as PyTorch tensors.
        device: The hardware device (CPU or GPU) for computation.
        batch_size (int): The number of documents to process in a batch.

    Returns:
        torch.Tensor: A single tensor containing concatenated embeddings.
    """
    embeddings = []

    # Batch processing for efficiency
    for i in tqdm(range(0, len(tokenized_docs["input_ids"]), batch_size)):
        batch = {k: v[i:i + batch_size] for k, v in tokenized_docs.items()}

        # Disable gradient tracking for inference
        with torch.no_grad():
            embeddings.append(model.first_stage_model(**batch))

    # Concatenate the individual batched embeddings
    return torch.cat(embeddings)


def run_stage_1(corpus, model, tokenizer, device):
    """
    Orchestrates the overall process of sampling a corpus, tokenizing it, and generating embeddings.

    Args:
        corpus (dict): Dictionary of documents to process.
        model: ML model with a `first_stage_model` attribute.
        tokenizer: Tokenizer for processing text input to the model.
        device: Device where computations will run (CPU/GPU).

    Returns:
        torch.Tensor: A tensor of embeddings for the selected corpus.
    """

    # Step 1: Sample a "mini-corpus" based on the model configuration
    minicorpus_docs = get_minicorpus(corpus, size=model.config.transductive_corpus_size)

    # Step 2: Tokenize the sampled documents
    tokenized_docs = tokenize(tokenizer, minicorpus_docs, device, DOCUMENT_PREFIX)

    # Step 3: Move model and tokenized data to the designated device
    model.to(device)

    # Step 4: Generate embeddings using the model
    embeddings = generate_embeddings(model, tokenized_docs, device)

    return embeddings
