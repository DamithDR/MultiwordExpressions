import pandas as pd

df_test = pd.read_csv('examples/mwe/en/data/mwe/processed/spanish/spanish_test.tsv', sep='\t')
print(len(df_test))
# i = 0
# with open('examples/mwe/en/data/mwe/processed/spanish/test.txt', 'r') as f:
#     content = f.read()
#
#     for word in content.split(' '):
#         if word.__eq__('nan'):
#             print('A')
#             line = pd.DataFrame({"words": "-", "labels": "O", "sentence_id": df_test.at[i, 'sentence_id']}, index=[i])
#             df_test = pd.concat([df_test.iloc[:i], line, df_test.iloc[i:]]).reset_index(drop=True)
#         i += 1
#
# print(len(df_test))

sent_ids = set()
sentence = ''
sentence_list = []
cur_id = 0
for word, sent_no in zip(df_test['words'], df_test['sentence_id']):
    if sent_no not in sent_ids :
        sent_ids.add(sent_no)
    if cur_id != sent_no:
        sentence_list.append(sentence + '\n')
        sentence = ''
    if sentence == '':
        sentence = word
    else:
        sentence += " " + word
    cur_id = sent_no

with open('change_test.txt','w') as f:
    f.writelines(sentence_list,)
print(sentence_list)
