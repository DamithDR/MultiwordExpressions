import time

import openai

openai.api_key = input("Please enter your OpenAI API key")

with open('examples/mwe/en/data/metaphoric/processed/test.txt', 'r') as file:
    sentences = file.readlines()

responses = []
# 217,255,304
for sentence in sentences[304:]:
    message = "is there a metaphoric flower name or metaphoric plant name included in the following sentence, say yes or no, if yes what is the metaphoric flower or metaphoric plant names in the sentence seperately : " + sentence
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": message}
        ]
    )
    resp = response['choices'][0]['message']['content']
    resp = str(resp).replace('\n', '##')
    responses.append(resp)
    with open('resp.txt', 'a') as f:
        f.write(str(resp).replace('\n', '##') + '\n')
    print(resp)
    time.sleep(3)  # free version only allows 20requests/min

print('Done')
