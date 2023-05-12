import sys
import time

import openai
import tiktoken as tiktoken

# openai.api_key = input("Please enter your OpenAI API key : ")
openai.api_key = "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"

with open('examples/mwe/en/data/mwe/processed/test.txt', 'r') as file:
    sentences = file.readlines()
# with open('examples/mwe/en/data/metaphoric/processed/spanish/test.txt', 'r') as file:
#     sentences = file.readlines()

# 217,255,304

# sentences = ['Whitefly and red spider mite may be troublesome bleeding heart']

for sentence in sentences[320:]:
    sentence = sentence.replace('\n', '')
    message = f"""
                - Find whether is there a multi-word expression flower or plant name in the text delimited by ```
                - if there is no multi-word expression found in the given text; just tell 'No' 
                - if you find a multiword expression in the given text; say yes and then give the multiword flower or plant name for example; Yes - 'Name' 
                  Text : ```{sentence}```
                """
    print(message)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "user", "content": message}
        ]
    )
    resp = response['choices'][0]['message']['content']
    resp = str(resp).replace('##', '\n')
    with open('mwe-english-resp.txt', 'a') as f:
        f.write(str(resp).replace('\n', '##') + '\n')
    print(resp)
    time.sleep(20)

# print('Done')
