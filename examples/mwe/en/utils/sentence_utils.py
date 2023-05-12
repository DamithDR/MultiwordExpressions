import numpy as np


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


def set_sequence_and_clean(df):
    seq = np.zeros(len(df.index), dtype='i4')
    df['sentence_id'] = seq

    cur_sid = 1
    lst = df.index[df['token_id'] == '1'].to_list()
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