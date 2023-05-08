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
df_test = pd.read_csv('examples/mwe/en/data/metaphoric/processed/spanish/spanish_test.tsv', sep='\t')
with open('examples/mwe/en/data/metaphoric/processed/spanish/test.txt', 'r') as f:
    test_sentences = f.readlines()

with open('spanish-resp.txt', 'r') as f:
    responses = f.readlines()

regex_list = [
    r"Yes, the metaphoric flower in the sentence is \"(?P<flower_name>[a-zA-Z ]+).",
    r"Yes, the metaphoric plant name is \"(?P<plant_name>[a-zA-Z ]+)",
    r"Yes - the metaphoric plant name is (?P<plant_name>[a-zA-Z ']+).",
    r"Yes - Metaphoric flower name: (?P<flower_name>[a-zA-Z ]+)",
    r"Yes. Metaphoric plant name: (?P<plant_name>[a-zA-Z ]+) Metaphoric flower name: (?P<flower_name>[a-zA-Z ]+)",
    r"Yes, the metaphoric flower name is \"(?P<flower_name>[a-zA-Z ]+).",
    r"Yes, the metaphoric plant name is \"(?P<plant_name>[a-zA-Z ]+)",
    r"Yes, the metaphoric plant name is (?P<plant_name>[a-zA-Z ]+)",
    r"Yes. Metaphoric flower/plant name: \"(?P<plant_name>[a-zA-Z ]+)",
    r"Yes, the metaphoric plant name in the sentence is \"(?P<plant_name>[a-zA-Z ]+)",
    r"Yes, the metaphoric plant name in the sentence is (?P<plant_name>[a-zA-Z ]+)",
    r"Yes, the metaphoric flower name in the sentence is \"(?P<flower_name>[a-zA-Z ]+)",
    r"Yes, the metaphoric flower name in the sentence is (?P<flower_name>[a-zA-Z ]+)",
    r"Yes, the metaphoric flower or plant name is  \"(?P<flower_name>[a-zA-Z ]+)",
    r"YesMetaphoric flower name: (?P<flower_name>[a-zA-Z ]+)",
    r"Yes, the metaphoric plant name is \"(?P<plant_name>[a-zA-Z ]+)\"",
    r"Yes, the metaphoric flower name in the sentence is \"(?P<flower_name>[a-zA-Z ,]+)",
    r"Yes, the metaphoric plant name in the sentence is \"(?P<plant_name>[a-zA-Z ,]+)",
    r"Yes, the metaphoric plant name is \"(?P<plant_name>[a-zA-Z ,]+)",
    r"Yes, the metaphoric flower name is \"(?P<flower_name>[a-zA-Z ]+),",
    r"Yes. Metaphoric Plant Names: (?P<flower_name>[a-zA-Z ]+).",
    r"Yes. (?P<flower_name>[a-zA-Z ]+) is a metaphorical flower name.",
    r"Yes, (?P<flower_name>[a-zA-Z ]+) is a metaphorical flower name.",
    r"Yes. Metaphoric plant names: (?P<flower_name>[a-zA-Z ]+),",
    r"Yes. The metaphoric flower name in the sentence is \"(?P<flower_name>[a-zA-Z0-9 -]+),",
    r"Yes, the metaphoric flower or plant name is \"(?P<flower_name>[a-zA-Z0-9 -]+)",
    r"Yes, the metaphoric flower name is (?P<flower_name>[a-zA-Z0-9 -]+').",
    r"Yes. Metaphoric plant name: (?P<plant_name>[a-zA-Z0-9 -]+)",
    r"Yes, \"(?P<plant_name>[a-zA-Z0-9 -]+)\" is a metaphoric plant name.",
    r"Yes. Metaphoric flower/plant name: (?P<plant_name>[a-zA-Z ']+).",
    r"Yes. the metaphoric flower/plant name in the sentence is (?P<plant_name>[a-zA-Z ']+).",
    r"Yes. Metaphoric plant name - (?P<plant_name>[a-zA-Z ']+).",
    r"Yes, the metaphoric plant names in the sentence are \"(?P<plant_name>[a-zA-Z ']+)",
    r"Yes, \"(?P<plant_name>[a-zA-Z ']+)\" could be considered a metaphoric plant name",
    r"Yes, the metaphoric name is \"(?P<plant_name>[a-zA-Z ']+)\" which could be interpreted as a",
    r"Yes, the metaphorical flower name in the sentence is \"(?P<plant_name>[a-zA-Z ']+)\"",
    r"Yes. Metaphoric flower name: (?P<plant_name>[a-zA-Z ']+) \(which means ",
    r"Yes, there are two metaphoric flower names in the sentence: 1. \"(?P<plant_name>[a-zA-Z -]+)\" and \"(?P<flower_name>[a-zA-Z ']+)\"",
    r"Yes. Metaphoric plant name: \'(?P<plant_name>[a-zA-Z -]+)\'.",
    r"Yes. Metaphoric plant name: \"(?P<plant_name>[a-zA-Z -]+)\" refers to",
    r"Yes. The metaphoric flower name is \"(?P<flower_name>[a-zA-Z -]+)\".",
    r"Yes, the metaphoric flower name in the sentence is \'(?P<flower_name>[a-zA-Z -]+)\'.",
    r"Yes. Metaphoric flower name: (?P<flower_name>[a-zA-Z -]+)",
    r"Yes. Metaphoric flower name: \"(?P<flower_name>[a-zA-Z -]+)\"",
    r"Yes - The metaphoric plant name is \"(?P<flower_name>[a-zA-Z -]+)\"",
    r"Yes, the metaphoric flower name is (?P<flower_name>[a-zA-Z -]+).",
    r"Yes - the metaphoric flower name is \"(?P<flower_name>[a-zA-Z -]+)\",",
    r"Yes, the metaphoric flower name in this sentence is \'(?P<flower_name>[a-zA-Z -]+)\',",
    r"Yes - The metaphoric flower name in the sentence is \"(?P<flower_name>[a-zA-Z ‘’]+)\"",
    r"Yes, the metaphoric plant names are \"(?P<plant_name>[a-zA-Z ]+)\" and ",
    r"Yes - metaphoric flower name: (?P<flower_name>[a-zA-Z ]+)",
    r"Yes, there are two metaphoric plant names in the sentence:1. \"(?P<plant_name>[a-zA-Z ]+)\"",
    r"Yes. Metaphoric plant name: \"(?P<plant_name>[a-zA-Z .]+)\"",
    r"Yes. The metaphoric flower name is \"(?P<flower_name>[a-zA-Z -]+),\" as it",
    r"Yes, there are two metaphoric plant names included in the given sentence: \"(?P<plant_name>[a-zA-Z\u0080-\uFFFF -]+)\" and \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -]+)\".",
    r"Yes, the metaphoric flower name included in the sentence is \"(?P<plant_name>[a-zA-Z\u0080-\uFFFF -]+)\"",
    r"Yes, there is a metaphoric plant name in the sentence - \"(?P<plant_name>[a-zA-Z\u0080-\uFFFF -]+)\".",
    r"Yes, there is a metaphoric plant name included in the sentence, which is \"(?P<plant_name>[a-zA-Z\u0080-\uFFFF -]+)\".",
    r"Yes, there is a metaphoric plant name in the sentence, which is \"(?P<plant_name>[a-zA-Z\u0080-\uFFFF -]+)\".",
    r"Yes, the sentence includes a metaphoric plant name \"(?P<plant_name>[a-zA-Z\u0080-\uFFFF -]+)\" which means",
    r"Yes, there is a metaphoric flower name in the sentence, which is \"(?P<plant_name>[a-zA-Z\u0080-\uFFFF -]+)\".",
    r"Yes, there are metaphoric plant names in the sentence. The metaphoric plant names are \"(?P<plant_name>[a-zA-Z\u0080-\uFFFF -]+)\" and \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -]+)\".",
    r"Yes, the metaphoric plant name in the sentence is \"(?P<plant_name>[a-zA-Z\u0080-\uFFFF -]+)\", which translates to",
    r"Yes, The metaphoric plant name is \"(?P<plant_name>[a-zA-Z\u0080-\uFFFF -,]+)\" which literally translates to",
    r"Yes, The metaphoric plant name is \"(?P<plant_name>[a-zA-Z\u0080-\uFFFF -,]+)\" which translates to",
    r"Yes, The metaphoric plant name is \"(?P<plant_name>[a-zA-Z\u0080-\uFFFF -,]+)\" which is a common",
    r"Yes, There is a metaphoric plant name included in the sentence: \"(?P<plant_name>[a-zA-Z\u0080-\uFFFF -,]+)\".",
    r"Yes, \"(?P<plant_name>[a-zA-Z\u0080-\uFFFF -,]+)\" \(vegetable ivory\) is a metaphoric plant name.",
    r"Yes, there is a metaphoric plant name included in the sentence - \"(?P<plant_name>[a-zA-Z\u0080-\uFFFF -,]+)\"",
    r"Yes, (?P<plant_name>[a-zA-Z\u0080-\uFFFF -,]+) and (?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+)",
    r"Yes, there are several metaphoric plant names in the sentence: \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+)\", \"(?P<plant_name>[a-zA-Z\u0080-\uFFFF -,]+)\",",
    r"Yes, the metaphoric plant name included in the sentence is \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+)\".",
    r"Yes, there is a metaphoric plant name in the sentence. The (?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+) can be considered a metaphoric plant name.",
    r"Yes, the sentence includes a metaphoric plant name: \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+)\" refers to",
    r"Yes, there is a metaphoric plant name included in the sentence: \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+)\".",
    r"Yes, The metaphoric plant name is \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+)\" which means ",
    r"Yes, there is a metaphoric plant name included in the sentence which is \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+)\"",
    r"Yes, there are two metaphoric plant names in the sentence: \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+)\"",
    r"Yes, the metaphoric plant name in the sentence is \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+)\"",
    r"Yes, \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+)\" is a metaphoric plant name,",
    r"Yes, the metaphoric plant name is \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+)\"",
    r"Yes, there is a metaphoric plant name included in the sentence. The plant is commonly known as \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+)\" which means",
    r"Yes, there are metaphoric plant names in the sentence: \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+)\" and \"(?P<plant_name>[a-zA-Z\u0080-\uFFFF -,]+)\"",
    r"Yes, \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+)\" can be considered a metaphoric plant name as it",
    r"Yes, there is a metaphoric flower name in the sentence. The metaphoric flower name is \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+)\" which means",
    r"Yes, \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+)\" \(wine palm\) is a",
    r"Yes, there is a metaphoric flower name included in the sentence. The metaphoric flower name is \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+)\" which means",
    r"Yes, there is a metaphoric plant name in the sentence which is \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+)\".",
    r"Yes, there is a metaphoric plant name in the sentence: \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+)\" \(Sacred fig\)",
    r"Yes, there is a metaphoric flower name in the sentence: (?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+).",
    r"Yes, there is a metaphoric plant name in the sentence. The metaphoric plant name is \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -,]+)\",",
    r"Yes, there is a metaphoric plant name included in the sentence, which is \"(?P<flower_name>[a-zA-Z\u0080-\uFFFF -,.]+)\"",
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

with open('not_captured_spanish.txt','w')as f:
    f.writelines(no_regex_list)
print(no_regex_list)

print(f'no of tokens {len(final_tag_list)}')

print(f'tokens in test set = {len(df_test["labels"])}')

with open('chatgpt-metaphoric-results-spanish.txt', 'w') as f:
    f.write(
        metrics.classification_report(df_test['labels'].tolist(), [tag for lst in final_tag_list for tag in lst],
                                      digits=6))
