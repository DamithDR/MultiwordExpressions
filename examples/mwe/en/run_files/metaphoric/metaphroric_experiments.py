import argparse

import pandas as pd
import torch
from sklearn import metrics

from mwe.transformers.ner_model import NERModel

parser = argparse.ArgumentParser(
    description='''evaluates models on metaphoric names identification task''')
parser.add_argument('--model_type', required=True, help='model type')
parser.add_argument('--model_name', required=True, help='model name')
parser.add_argument('--cuda_device', required=False, help='cuda_device', default=0)
args = parser.parse_args()

df_train = pd.read_csv('examples/mwe/en/data/metaphoric/processed/train.tsv', sep='\t')
df_test = pd.read_csv('examples/mwe/en/data/metaphoric/processed/test.tsv', sep='\t')
test_sentences = []
with open('examples/mwe/en/data/metaphoric/processed/test.txt', 'r') as f:
    test_sentences = f.readlines()

model = NERModel(
    model_type=args.model_type,
    model_name=args.model_name,
    labels=["O", "B", "I"],
    use_cuda=torch.cuda.is_available(),
    cuda_device=args.cuda_device,
    args={"overwrite_output_dir": True,
          "reprocess_input_data": True,
          "num_train_epochs": 3,
          "train_batch_size": 32,
          # "wandb_project": 'metaphoric_flowers',
          "use_multiprocessing": False,
          "use_multiprocessing_for_evaluation": False
          },
)

# Train the model
model.train_model(df_train)

result, model_outputs, preds_list = model.eval_model(df_test)

with open(str(args.model_name).replace('/', '-') + 'results.txt', 'w') as f:
    f.write(
        metrics.classification_report(df_test['labels'].tolist(), [tag for lst in preds_list for tag in lst], digits=4))
