from gensim.models import word2vec
import gensim
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

story = open("Input.txt", "r")
s = story.read()

s = s.replace("\n", " ")

relations = []

for i in sent_tokenize(s):
    temp = []
    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())

    relations.append(temp)

print(relations)