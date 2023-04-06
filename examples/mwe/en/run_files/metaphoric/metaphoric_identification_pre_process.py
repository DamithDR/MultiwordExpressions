import sys

import numpy as np
import pandas as pd

from examples.mwe.en.utils.sentence_utils import get_sentences, set_sequence_and_clean

train_f = 'examples/mwe/en/data/metaphoric/raw/spanish_train.tsv'
test_f = 'examples/mwe/en/data/metaphoric/raw/spanish_test.tsv'

train_df = pd.read_csv(train_f, sep='\t')
test_df = pd.read_csv(test_f, sep='\t')

df_train_cleaned = set_sequence_and_clean(train_df)
df_gold_cleaned = set_sequence_and_clean(test_df)

train_sentences, train_tokens = get_sentences(train_df)
test_sentences, test_tokens = get_sentences(test_df)

all_sentences = train_sentences + test_sentences
all_tokens = train_tokens + test_tokens

bi_sentences = []
bi_tokens = []
o_sentences = []
o_tokens = []

for tok_list, sent in zip(all_tokens, all_sentences):
    if 'B' not in tok_list and 'I' not in tok_list:
        o_sentences.append(sent)
        o_tokens.append(tok_list)
    else:
        bi_sentences.append(sent)
        bi_tokens.append(tok_list)


print(len(all_sentences))
print(len(o_sentences))
print(len(bi_sentences))
sys.exit(0)
# check the number of sentences here

o_sentences = o_sentences[:101]
o_tokens = o_tokens[:101]

bi_sentences += o_sentences
bi_tokens += o_tokens

sent_arr = np.array(bi_sentences)
tok_arr = np.array(bi_tokens)

# shuffle the indexes
r_indexes = np.arange(len(sent_arr))
np.random.shuffle(r_indexes)

sent_arr = sent_arr[r_indexes]
tok_arr = tok_arr[r_indexes]

sentence_list = sent_arr.tolist()
token_list = tok_arr.tolist()

print(f'total number of sentences : {len(sentence_list)}')
print(f'total number of tokens : {len(token_list)}')

sentence_train = sentence_list[0:1500]
sentence_test = sentence_list[1500:2020]

token_train = token_list[0:1500]
token_test = token_list[1500:2020]

training_df = pd.DataFrame()
training_df['words'] = [word for sentence in sentence_train for word in sentence.split(' ')]
training_df['labels'] = [tag for tk_lst in token_train for tag in tk_lst]
sentence_wise_ids_train = [[i] * len(sentence_train[i].split(' ')) for i in range(0, len(sentence_train))]
training_df['sentence_id'] = [id for sent in sentence_wise_ids_train for id in sent]

test_df = pd.DataFrame()
test_df['words'] = [word for sentence in sentence_test for word in sentence.split(' ')]
test_df['labels'] = [tag for tk_lst in token_test for tag in tk_lst]
sentence_wise_ids_test = [[i] * len(sentence_test[i].split(' ')) for i in range(0, len(sentence_test))]
test_df['sentence_id'] = [id for sent in sentence_wise_ids_test for id in sent]

training_df.to_csv('examples/mwe/en/data/metaphoric/processed/spanish_test.tsv', sep='\t', index=False)
test_df.to_csv('examples/mwe/en/data/metaphoric/processed/spanish_test.tsv', sep='\t', index=False)
with open('examples/mwe/en/data/metaphoric/processed/test.txt', 'w') as f:
    s_lst = '\n'.join(sentence_test)
    f.write(s_lst)

# total training 38986 - only O - 37866
# total testing 14233 - Only O - 13933
