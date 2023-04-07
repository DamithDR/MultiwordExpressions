import argparse

import pandas as pd
import torch
from sklearn import metrics

from mwe.transformers.ner_model import NERModel

parser = argparse.ArgumentParser(
    description='''evaluates models on metaphoric names identification task''')
parser.add_argument('--model_type', required=False, help='model type')
parser.add_argument('--model_name', required=False, help='model name')
parser.add_argument('--cuda_device', required=False, help='cuda_device', default=0)
args = parser.parse_args()

df_train = pd.read_csv('examples/mwe/en/data/metaphoric/processed/train.tsv', sep='\t')
df_test = pd.read_csv('examples/mwe/en/data/metaphoric/processed/spanish/spanish_test.tsv', sep='\t')
test_sentences = []
with open('examples/mwe/en/data/metaphoric/processed/spanish/test.txt', 'r') as f:
    test_sentences = f.readlines()

if args.model_name is None or args.model_type is None:

    model_names_list = ['xlm-roberta-base', 'xlm-roberta-large',
                        'bert-base-multilingual-cased', 'bert-base-multilingual-uncased']
    # model_names_list = ['bert-base-multilingual-cased', 'bert-base-multilingual-uncased']
    model_types_list = ['xlmroberta', 'xlmroberta', 'bert', 'bert']
    # model_types_list = ['bert', 'bert']
else:
    model_names_list = [args.model_name]
    model_types_list = [args.model_type]

for model_name, model_type in zip(model_names_list, model_types_list):
    print(f'running experiment on {model_name}')

    model = NERModel(
        model_type=model_type,
        model_name=model_name,
        labels=["O", "B", "I"],
        use_cuda=torch.cuda.is_available(),
        cuda_device=args.cuda_device,
        args={"overwrite_output_dir": True,
              "reprocess_input_data": True,
              "num_train_epochs": 3,
              "train_batch_size": 32,
              "use_multiprocessing": False,
              "use_multiprocessing_for_evaluation": False
              },
    )

    # Train the model
    model.train_model(df_train)

    n_list = df_test.loc[df_test['sentence_id'] == '']

    result, model_outputs, preds_list, truth, preds = model.eval_model(df_test)
    with open('metaphoricresults/' + str(model_name).replace('/', '-') + '-results-spanish.txt', 'w') as f:
        f.write(
            metrics.classification_report(truth, preds, digits=6))
