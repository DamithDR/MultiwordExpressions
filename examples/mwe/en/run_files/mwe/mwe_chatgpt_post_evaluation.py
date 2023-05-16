import re
import sys

import pandas as pd
from sklearn import metrics
# df_train = pd.read_csv('examples/mwe/en/data/metaphoric/processed/spanish_test.tsv', sep='\t')
# df_test = pd.read_csv('examples/mwe/en/data/metaphoric/processed/spanish_test.tsv', sep='\t')
# with open('examples/mwe/en/data/metaphoric/processed/test.txt', 'r') as f:
#     test_sentences = f.readlines()
#
# with open('examples/mwe/en/responses/metaphoric/chatgpt_reponses.txt', 'r') as f:
#     responses = f.readlines()


# df_train = pd.read_csv('examples/mwe/en/data/metaphoric/processed/spanish_test.tsv', sep='\t')
df_test = pd.read_csv('examples/mwe/en/data/mwe/processed/spanish/spanish_test.tsv', sep='\t')
with open('examples/mwe/en/data/mwe/processed/spanish/test.txt', 'r') as f:
    test_sentences = f.readlines()

with open('mwe-spanish-resp.txt', 'r') as f:
    responses = f.readlines()

regex_list = [
    r"Yes, the metaphoric flower in the sentence is \"(?P<flower_name>[a-zA-Z ]+).",
    r"Yes - (?P<flower_name>[a-zA-Z 'Á]+)",
]

# m = re.match(r"(?P<first_name>\w+) ABC (?P<last_name>\w+)", "Malcom ABC Reynolds sbc")
# group = m.groupdict()
# print(group)

# text = "Yes, the metaphoric plant name is \"rosetted,\""
#
# match = re.match(r"Yes, the metaphoric plant name is \"(?P<flower_name>[a-zA-Z ]+),", text)
# group = match.groupdict()
# print(group)


final_tag_list = []
lists = []
no_regex_list = []
for sentence, response in zip(test_sentences, responses):
    response = response.replace('##', '')
    if response.lower().startswith('no'):
        final_tag_list.extend(['O'] * len(sentence.split(' ')))
    else:
        tags=[]
        if response.lower().startswith('yes'):
            matched = False
            for regex in regex_list:
                match = re.match(regex, response)
                if match is not None:
                    matched = True
                    group = match.groupdict()
                    flower_name = group.get('flower_name') if group.get('flower_name') is not None else ''
                    plant_name = group.get('plant_name') if group.get('plant_name') is not None else ''
                    cur_tag = 'O'
                    split_list = sentence.split(' ')
                    tags = ['O'] * len(sentence.split(' '))
                    for i in range(len(split_list)):
                        word = split_list[i].replace('‘', '').replace(',','').replace('\'','').replace('’','')
                        if flower_name.__contains__(word) and cur_tag == 'O':
                            tags[i] = 'B'
                            cur_tag = 'B'
                        elif flower_name.__contains__(word) and cur_tag == 'B':
                            tags[i]='I'
                            cur_tag = 'I'
                        elif flower_name.__contains__(word) and cur_tag == 'I':
                            tags[i]='I'
                            cur_tag = 'I'

                        if plant_name.__contains__(word) and cur_tag == 'O':
                            tags[i]='B'
                            cur_tag = 'B'
                        elif plant_name.__contains__(word) and cur_tag == 'B':
                            tags[i]='I'
                            cur_tag = 'I'
                        elif plant_name.__contains__(word) and cur_tag == 'I':
                            tags[i]='I'
                            cur_tag = 'I'
                    final_tag_list.extend(tags)
                    break

            if not matched:
                no_regex_list.append(response)

print(len(no_regex_list))
with open('not_captured_mwe_english.txt','w')as f:
    f.writelines(no_regex_list)
print(no_regex_list)

print(f'no of tokens {len(final_tag_list)}')

print(f'tokens in test set = {len(df_test["labels"])}')

with open('chatgpt-mwe-spanish-results.txt', 'w') as f:
    f.write(
        metrics.classification_report(df_test['labels'].tolist(), [tag for lst in final_tag_list for tag in lst],
                                      digits=6))
