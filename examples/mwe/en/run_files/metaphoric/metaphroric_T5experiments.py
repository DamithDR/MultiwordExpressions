import argparse
import json

import pandas as pd
import torch
from sklearn import metrics

from mwe.t5.t5_model import T5Model
from mwe.transformers.config.model_args import T5Args
from mwe.transformers.ner_model import NERModel

parser = argparse.ArgumentParser(
    description='''evaluates models on metaphoric names identification task''')
parser.add_argument('--model_type', required=False, help='model type')
parser.add_argument('--model_name', required=False, help='model name')
parser.add_argument('--epochs', required=False, help='no of epochs')
parser.add_argument('--cuda_device', required=False, help='cuda_device', default=0)
args = parser.parse_args()

df_train = pd.read_csv('examples/mwe/en/data/metaphoric/processed/train.tsv', sep='\t')
df_test = pd.read_csv('examples/mwe/en/data/metaphoric/processed/test.tsv', sep='\t')
with open('examples/mwe/en/data/metaphoric/processed/test.txt', 'r') as f:
    test_sentences = f.readlines()

train_sentence_ids = set(df_train['sentence_id'].tolist())
test_sentence_ids = set(df_test['sentence_id'].tolist())
TASK_NAME = 'token classification'

train_data = []
test_data = []

for train_id in train_sentence_ids:
    subset = df_train.loc[df_train['sentence_id'] == train_id]
    train_data.append([TASK_NAME, ' '.join(subset['words'].tolist()), ' '.join(subset['labels'].tolist())])

for test_id in test_sentence_ids:
    subset = df_test.loc[df_test['sentence_id'] == test_id]
    test_data.append(TASK_NAME + ": " ' '.join(subset['words'].tolist()))

train_df = pd.DataFrame(train_data)
train_df.columns = ["prefix", "input_text", "target_text"]

# Configure the model
model_args = T5Args()
model_args.num_train_epochs = int(args.epochs)
model_args.no_save = True
model_args.overwrite_output_dir = True
model_args.use_multiprocessing=False
model_args.use_multiprocessing_for_evaluation=False
model_args.use_multiprocessed_decoding=False
model_args.multiprocessing_chunksize=False

model = T5Model(args.model_type, args.model_name, args=model_args)

# Train the model
model.train_model(train_df)

preds_list = model.predict(test_data)
print(preds_list)

with open('metaphoricresults/' + str(args.model_name).replace('/', '-') + '-results.txt', 'w') as f:
    f.write(
        metrics.classification_report(df_test['labels'].tolist(), [tag for lst in preds_list for tag in lst.split(' ')],
                                      digits=6))
