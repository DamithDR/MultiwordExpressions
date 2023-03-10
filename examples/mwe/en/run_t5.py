import os
import pandas as pd
import torch

from examples.mwe.en.run import set_sequence_and_clean_t5, get_sentences_t5
from mwe.t5.t5_model import T5Model
from mwe.transformers.util.print_stat import print_information_multi_class

MULTI_LABEL_CLASSIFICATION = "multilabel classification"

model_args = {
        "max_seq_length": 196,
        "train_batch_size": 16,
        "eval_batch_size": 64,
        "num_train_epochs": 3,
        "evaluate_during_training": False,
        "evaluate_during_training_steps": 15000,
        "evaluate_during_training_verbose": True,

        "use_multiprocessing": False,
        "fp16": False,

        "save_steps": -1,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": False,

        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "wandb_project": "Multi-Label-Classification",
    }

def t5_based_evaluation():
    print("T5 Model Based Predictions")

    train_f = os.path.join(".", "examples", "mwe", "en", "data_backup", "dimsum16.train")
    gold_test_f = os.path.join(".", "examples", "mwe", "en", "data_backup", "dimsum16.test")

    df_train = pd.read_csv(train_f, usecols=[0, 1, 4], names=["token_id", "input_text", "target_text"], sep='\t')
    df_train = set_sequence_and_clean_t5(df_train)
    df_train["prefix"] = MULTI_LABEL_CLASSIFICATION
    df_gold = pd.read_csv(gold_test_f, usecols=[0, 1, 4], names=["token_id", "input_text", "target_text"], sep='\t')
    df_gold = set_sequence_and_clean_t5(df_gold)
    df_gold["prefix"] = MULTI_LABEL_CLASSIFICATION

    model = T5Model("t5", "t5-base", args=model_args, use_cuda=torch.cuda.is_available())

    model.train_model(df_train)

    sentences, tokens = get_sentences_t5(df_gold)
    tokens_to_predict = []
    for sent in sentences:
        for token in sent.split(" "):
            tokens_to_predict.append(token)

    predictions = model.predict(tokens_to_predict)

    df_gold['predicted'] = predictions

    print_information_multi_class(df_gold, "predicted", "target_text")


if __name__ == '__main__':
    t5_based_evaluation()
