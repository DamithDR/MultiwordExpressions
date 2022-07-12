from nltk import ngrams

sent = "test sent"
sent1 = "test sent is this"
tets = "is this"
print(sent1.rindex(tets))


grams =ngrams(sent.split(),3)

