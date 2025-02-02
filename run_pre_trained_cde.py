import argparse
from data_load.get_dataset import get_dataset
from data_load.get_split import GetSplit
from pre_trained_cde.run_stage_1 import run_stage_1
from pre_trained_cde.run_stage_2 import run_stage_2
from pre_trained_cde.save_pre_trained import save
import transformers
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="BEIR dataset name.")

args = parser.parse_args()

dataset_name = args.dataset

get_dataset(dataset_name)

get_split = GetSplit(dataset_name)

model = transformers.AutoModel.from_pretrained("jxm/cde-small-v1", trust_remote_code=True)
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stage_1_embeddings = run_stage_1(get_split.train[0], model, tokenizer, device)

docs_tensor, train_tensor, dev_tensor, test_tensor = run_stage_2(get_split.train[1], model, tokenizer,
                                                                 device, stage_1_embeddings, get_split)

save(docs_tensor, train_tensor, dev_tensor, test_tensor, dataset_name)
