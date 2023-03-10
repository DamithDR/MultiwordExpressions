import json

from bi_lstm_crf.app import WordsTagger
import os
import pandas as pd

from examples.mwe.en.run import set_sequence_and_clean, get_sentences
from mwe.transformers.util.print_stat import print_information_multi_class

model = WordsTagger(model_dir='lstm-crf-model', device='cpu')

gold_test_f = os.path.join('.', 'data_backup', 'flower_dataset_all_final_test.tsv')
df_gold = pd.read_csv(gold_test_f, sep='\t')
df_gold = set_sequence_and_clean(df_gold)
sents, toks = get_sentences(df_gold)
predictions = []
for sent in sents:
    split = sent.split(' ')
    tags, sequences = model([split])
    predictions += tags[0]

df_gold['predicted'] = predictions
print_information_multi_class(df_gold, "predicted", "labels")


