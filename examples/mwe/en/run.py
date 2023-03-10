import argparse
import os

import numpy as np
import pandas as pd
import torch

from examples.mwe.en.utils.Args import Args
from mwe.transformers.ner_model import NERModel
from mwe.transformers.util.print_stat import print_information_multi_class


def set_sequence_and_clean(df):
    seq = np.zeros(len(df.index), dtype='i4')
    df['sentence_id'] = seq

    cur_sid = 1
    lst = df.index[df['token_id'] == 1].to_list()
    size = len(lst)
    for i in range(0, size):
        if i + 1 < size:
            from_n = lst[i]
            to_n = lst[i + 1]
            for j in range(from_n, to_n):
                df.loc[j, 'sentence_id'] = cur_sid
        else:
            from_n = lst[i]
            for j in range(from_n, len(df.index)):
                df.loc[j, 'sentence_id'] = cur_sid
        cur_sid += 1
    df['labels'] = df['labels'].str.upper()
    df = df.drop(df[df['labels'] == '0'].index)
    return df


def set_sequence_and_clean_t5(df):
    seq = np.zeros(len(df.index), dtype='i4')
    df['sentence_id'] = seq

    cur_sid = 1
    lst = df.index[df['token_id'] == 1].to_list()
    size = len(lst)
    for i in range(0, size):
        if i + 1 < size:
            from_n = lst[i]
            to_n = lst[i + 1]
            for j in range(from_n, to_n):
                df.loc[j, 'sentence_id'] = cur_sid
        else:
            from_n = lst[i]
            for j in range(from_n, len(df.index)):
                df.loc[j, 'sentence_id'] = cur_sid
        cur_sid += 1
    df['target_text'] = df['target_text'].str.upper()
    df = df.drop(df[df['target_text'] == '0'].index)
    return df


def get_sentences(df_gold):
    sentences = []
    tokens = []
    cur_sid = 1
    sub_list = []
    sub_list_token = []
    for i in range(0, len(df_gold.index)):
        if cur_sid == df_gold.iloc[i]['sentence_id']:
            sub_list.append(str(df_gold.iloc[i]['words']))
            sub_list_token.append(df_gold.iloc[i]['labels'])
        else:
            sentences.append(" ".join(sub_list))
            tokens.append(sub_list_token)
            cur_sid += 1
            sub_list = [str(df_gold.iloc[i]['words'])]
            sub_list_token = [df_gold.iloc[i]['labels']]
        if i == len(df_gold.index) - 1:
            sentences.append(" ".join(sub_list))
            tokens.append(sub_list_token)
    return sentences, tokens

def get_sentences_t5(df_gold):
    sentences = []
    tokens = []
    cur_sid = 1
    sub_list = []
    sub_list_token = []
    for i in range(0, len(df_gold.index)):
        if cur_sid == df_gold.iloc[i]['sentence_id']:
            sub_list.append(str(df_gold.iloc[i]['input_text']))
            sub_list_token.append(df_gold.iloc[i]['target_text'])
        else:
            sentences.append(" ".join(sub_list))
            tokens.append(sub_list_token)
            cur_sid += 1
            sub_list = [str(df_gold.iloc[i]['input_text'])]
            sub_list_token = [df_gold.iloc[i]['target_text']]
        if i == len(df_gold.index) - 1:
            sentences.append(" ".join(sub_list))
            tokens.append(sub_list_token)
    return sentences, tokens

def bert_based_evaluation(model):
    train_f = os.path.join(".", "examples", "mwe", "en", "data_backup", "dimsum16.train")
    gold_test_f = os.path.join(".", "examples", "mwe", "en", "data_backup", "dimsum16.test")

    df_train = pd.read_csv(train_f, usecols=[0, 1, 4], names=["token_id", "words", "labels"], sep='\t')
    df_train = set_sequence_and_clean(df_train)
    df_gold = pd.read_csv(gold_test_f, usecols=[0, 1, 4], names=["token_id", "words", "labels"], sep='\t')
    df_gold = set_sequence_and_clean(df_gold)

    # used models
    model_types_dict = {
        "xlm-roberta-base": "xlmroberta",
        "xlnet-base-cased": "xlnet",
        "roberta-base": "roberta",
        "bert-base-multilingual-cased": "bert",
        "bert-base-multilingual-uncased": "bert",
        "bert-base-uncased": "bert",
        "bert-base-cased": "bert",
        "google/electra-base-discriminator": "electra"
    }
    print("Model detected : " + model)
    print("Model type detected : " + model_types_dict[model])
    model = NERModel(
        model_type=model_types_dict[model],
        model_name=model,
        labels=Args["tags"],
        use_cuda=torch.cuda.is_available(),
        args={"overwrite_output_dir": True,
              "reprocess_input_data": True,
              "num_train_epochs": 3,
              "train_batch_size": 32,
              },
    )

    # Train the model
    model.train_model(df_train)

    sentences, tokens = get_sentences(df_gold)
    predictions, raw_outputs = model.predict(sentences)

    predicted = []
    for p in predictions:
        for dic in p:
            predicted += list(dic.values())

    df_gold['predicted'] = predicted

    print_information_multi_class(df_gold, "predicted", "labels")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''evaluates models on multiword expression task''')
    parser.add_argument('--model', required=True, help='model')
    args = parser.parse_args()

    bert_based_evaluation(args.model)
