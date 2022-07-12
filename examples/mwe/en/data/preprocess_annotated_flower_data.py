import os.path
from nltk import ngrams
import pandas as pd

rows = []
rows_set = set()

if not os.path.exists('annotated_deduplicated.txt'):
    with open('annotated_deduplicated.txt', 'w', encoding='utf-8') as f:
        f.writelines(list(set(rows)))
    # print(len(set(rows)))

with open('annotated_deduplicated.txt', encoding='utf-8') as f:
    rows = f.readlines()
print(len(rows))

metaphorical_names = set()
not_metaphorical_names = set()
filtered_rows = []
full_string = ""
max_token_len = 0
for row_num in range(0, len(rows)):
    row = rows[row_num]
    split = row.split(',')
    if len(split) == 2:
        if len(split[0].split(' ')) > 1:
            if row.endswith('\n'):
                row = row[:len(row) - 1] + ", multiword expression\n"
            else:
                row += ", multiword expression\n"
        else:
            if row.endswith('\n'):
                row = row[:len(row) - 1] + ", not multiword expression\n"
            else:
                row += ", not multiword expression\n"
    if len(split[0].split(' ')) > max_token_len:
        print("change token len " + str(len(split[0].split(' '))))
        max_token_len = len(split[0].split(' '))

    if split[1].strip().lower() == "metaphorical":
        if not metaphorical_names.__contains__(split[0].strip().lower()):
            metaphorical_names.add(split[0].strip().lower())
            filtered_rows.append(row)
    elif split[1].strip().lower() == "not metaphorical":
        if not not_metaphorical_names.__contains__(split[0].strip().lower()):
            not_metaphorical_names.add(split[0].strip().lower())
            filtered_rows.append(row)

print("done")
with open('annotated_filtered.txt', 'w', encoding='utf-8') as f:
    rows = f.writelines(filtered_rows)
f.close()

with open('annotated_filtered.txt', encoding='utf-8') as f:
    rows = f.readlines()

multiword_names = set()
not_multiword_names = set()

for row_num in range(0, len(rows)):
    row = rows[row_num]
    split = row.split(',')

    if split[2].strip().lower() == "multiword expression":
        if not multiword_names.__contains__(split[0].strip().lower()):
            multiword_names.add(split[0].strip().lower())
    elif split[2].strip().lower() == "not multiword expression":
        if not not_multiword_names.__contains__(split[0].strip().lower()):
            not_multiword_names.add(split[0].strip().lower())

with open('flower_dataset_backup.txt', 'r', encoding='utf-8') as f:
    rows = f.readlines()
for row in rows:
    full_string += row.replace('\n', ' ')

sentences = full_string.split('.')

token_list = [[], [], [], [], [], []]
tag_list = [[], [], [], [], [], []]
tok_num_lst = [[], [], [], [], [], []]

multiword_exp_count = 0

for sent in sentences:
    print(sent)
    sent = sent.strip()  # added lately
    # if sent.isnumeric() or len(sent.strip()) < 2 or len(sent.strip().split(' ')) < 4 or (sent == '\n'):
    if sent.isnumeric() or len(sent.strip()) < 2 or len(sent.strip().split(' ')) < 2 or (sent == '\n'):
        print("--------------------------------------------")
        continue
    else:
        _2grams = ngrams(sent.split(), 2)
        _3grams = ngrams(sent.split(), 3)
        _4grams = ngrams(sent.split(), 4)
        _5grams = ngrams(sent.split(), 5)
        _6grams = ngrams(sent.split(), 6)
        _7grams = ngrams(sent.split(), 7)

    next_start = 0
    # n_grams_list = [_2grams, _3grams]
    n_grams_list = [_2grams, _3grams, _4grams, _5grams, _6grams, _7grams]
    for gram_id in range(0, len(n_grams_list)):
        token_count = gram_id + 2
        iterative_list = list(n_grams_list[gram_id])
        tok_num = 1
        if token_count > len(sent.strip().split(' ')):
            for split in sent.strip().split(' '):
                token_list[gram_id].append(split)
                tag_list[gram_id].append('O')
                tok_num_lst[gram_id].append(int(tok_num))
                tok_num += 1
            token_list[gram_id].append(None)
            tag_list[gram_id].append(None)
            tok_num_lst[gram_id].append(0)
            continue

        is_last_token_mwe = False
        final_tokens = []
        no_of_ngrams = len(list(n_grams_list[gram_id]))
        tokens_id_num = 0
        while tokens_id_num < len(iterative_list):

            # for tokens_id_num in range(0, len(iterative_list)):
            tokens = iterative_list[tokens_id_num]
            no_of_ngrams -= 1
            final_tokens = tokens
            expression = " ".join(tokens)
            if multiword_names.__contains__(expression):
                multiword_exp_count += 1
                is_last_token_mwe = True
                for tok_id in range(0, len(tokens)):
                    tok = tokens[tok_id]
                    token_list[gram_id].append(tok)
                    tok_num_lst[gram_id].append(int(tok_num))
                    if tok_id == 0:
                        tag_list[gram_id].append('B')
                    else:
                        tag_list[gram_id].append('I')
                    tok_num += 1
                # deduct_count = 1
                # if no_of_ngrams > token_count:
                #     for _ in range(token_count):
                #         tokens = next(iterator, None)
                leftover = sent[(sent.rindex(expression) + len(expression)):len(sent)]
                if token_count >= len(leftover.strip().split(' ')):
                    if not leftover.strip() == '':
                        for split in leftover.strip().split(' '):
                            token_list[gram_id].append(split)
                            tag_list[gram_id].append('O')
                            tok_num_lst[gram_id].append(int(tok_num))
                            tok_num += 1
                    # token_list[gram_id].append(None)
                    # tag_list[gram_id].append(None)
                    # tok_num_lst[gram_id].append(0)
                    break
                else:
                    iterative_list = list(ngrams(leftover.split(), token_count))
                    tokens_id_num = 0
            else:
                is_last_token_mwe = False
                token_list[gram_id].append(tokens[0])
                tok_num_lst[gram_id].append(int(tok_num))
                tag_list[gram_id].append('O')
                tok_num += 1
                tokens_id_num += 1
        if not is_last_token_mwe:
            for tk_id in range(1, len(final_tokens)):  # skipping the first element as it has considered already
                token_list[gram_id].append(final_tokens[tk_id])
                tag_list[gram_id].append('O')
                tok_num_lst[gram_id].append(int(tok_num))
                tok_num += 1
        token_list[gram_id].append(None)
        tag_list[gram_id].append(None)
        tok_num_lst[gram_id].append(0)

print("multiword expressions count = " + str(multiword_exp_count))
print("sentences count = " + str(len(sentences)))

final_tag_list = []
for i in range(0, len(token_list[0])):
    tag_lst1_out = tag_list[0][i]
    tag_lst2_out = tag_list[1][i]
    tag_lst3_out = tag_list[2][i]
    tag_lst4_out = tag_list[3][i]
    tag_lst5_out = tag_list[4][i]
    tag_lst6_out = tag_list[5][i]
    if tag_lst1_out == 'B' or tag_lst2_out == 'B' or tag_lst3_out == 'B' or tag_lst4_out == 'B' or tag_lst5_out == 'B' \
            or tag_lst6_out == 'B':
        final_tag_list.append('B')
    elif tag_lst1_out == 'I' or tag_lst2_out == 'I' or tag_lst3_out == 'I' or tag_lst4_out == 'I' or tag_lst5_out == 'I' \
            or tag_lst6_out == 'I':
        final_tag_list.append('I')
    elif tag_lst1_out is None:
        final_tag_list.append(None)
    else:
        final_tag_list.append('O')

dataset = pd.DataFrame()
dataset['sentence_id'] = tok_num_lst[0]
dataset.sentence_id.astype(int)
dataset['words'] = token_list[0]
dataset['labels'] = final_tag_list

dataset.to_csv('flower_dataset_all_final.tsv', sep='\t', index=False)

# dataset2 = pd.DataFrame()
# dataset2['sentence_id'] = tok_num_lst[1]
# dataset2.sentence_id.astype(int)
# dataset2['words'] = token_list[1]
# dataset2['labels'] = tag_list[1]
#
# dataset2.to_csv('flower_dataset3.tsv', sep='\t', index=False)
#
# dataset3 = pd.DataFrame()
# dataset3['sentence_id'] = tok_num_lst[2]
# dataset3.sentence_id.astype(int)
# dataset3['words'] = token_list[2]
# dataset3['labels'] = tag_list[2]
#
# dataset3.to_csv('flower_dataset4.tsv', sep='\t', index=False)

