import json
import os
import pandas as pd

from examples.mwe.en.run import set_sequence_and_clean, get_sentences
from examples.mwe.en.utils.Args import Args


def preprocess():
    train_f = os.path.join(".", "data", "flower_dataset_all_final_train.tsv")
    gold_test_f = os.path.join(".", "data", "flower_dataset_all_final_test.tsv")
    corpus_dir = os.path.join(".", "corpus_dir")

    df_train = pd.read_csv(train_f, sep='\t')
    df_train = set_sequence_and_clean(df_train)
    df_gold = pd.read_csv(gold_test_f, sep='\t')
    df_gold = set_sequence_and_clean(df_gold)

    vocab_list = list(set(df_train['words'] + df_gold['words']))
    vocab_list = list(map(str, vocab_list))
    tags = Args["tags"]
    vocab_json = json.dumps(vocab_list)
    tags_json = json.dumps(tags)

    sentences, tokens = get_sentences(df_train)
    with open(os.path.join(corpus_dir, "dataset.txt"), "w", encoding='utf-8') as text_file:
        for (sentence, token) in zip(sentences, tokens):
            text_file.write(json.dumps(list(sentence.split(' '))) + '\t' + json.dumps(token) + '\n')
    with open(os.path.join(corpus_dir, "vocab.json"), "w", encoding='utf-8') as vocab_file:
        vocab_file.write(vocab_json)
    with open(os.path.join(corpus_dir, "tags.json"), "w", encoding='utf-8') as tags_file:
        tags_file.write(tags_json)


if __name__ == '__main__':
    preprocess()
