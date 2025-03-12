import pickle
import torch
from beir.datasets.data_loader import GenericDataLoader
import argparse
from analysis.results_utils import calculate_map
import models.with_attention as with_attention
import models.more_positive as more_positive

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="BEIR dataset name.")
parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden Dimension For the Model")

args = parser.parse_args()

data_path = "datasets/" + args.dataset

# getting train queries, documents and qrels
corpus, queries_train, qrels_train = GenericDataLoader(data_path).load(split="train")
corpus_list = list(corpus.items())

with open(f'doc_embeddings_{args.dataset}.pkl', 'rb') as f:
    doc_embeddings_tensor = pickle.load(f)

with open(f'query_embeddings_train_{args.dataset}.pkl', 'rb') as f:
    train_query_embeddings_tensor = pickle.load(f)

_, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")

with open(f'query_embeddings_dev_{args.dataset}.pkl', 'rb') as f:
    test_query_embeddings_tensor = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_dim = 768
hidden_dim = args.hidden_dim
query_adaptive_model = with_attention.QueryAdaptiveCDE(embedding_dim, hidden_dim, num_heads=4)

# Define optimizer and loss function
optimizer = torch.optim.Adam(query_adaptive_model.parameters(), lr=1e-5)


for i in range(30):
    print(f"Train {i}")
    dataset = more_positive.QueryDataset(doc_embeddings_tensor, train_query_embeddings_tensor, list(queries_train.keys()),
                                          qrels_train, list(corpus.keys()), num_negatives=500, max_positives=5)
    dataloader = more_positive.DataLoader(dataset, batch_size=32, shuffle=True,
                                          collate_fn=more_positive.custom_collate_fn)
    more_positive.train_query_adaptive_model(query_adaptive_model, dataloader, optimizer, num_epochs=1)

    if (i + 1) % 1 == 0:

        test_map_score = calculate_map(query_adaptive_model, doc_embeddings_tensor, test_query_embeddings_tensor,
                                      list(test_queries.keys()), test_qrels, list(corpus.keys()))
        print(f"Mean Average Precision (MAP) on test set: {test_map_score:.4f}")

