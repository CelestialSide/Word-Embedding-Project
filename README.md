# Overview
This project intends to create a Word2Vec model from scratch with the goal of finding semetic relationships
between words in 1800s Gothic Literature.

# Dataset
We created a dataset from scratch for this project. In order to create the dataset we first pulled the raw text
of many books within our time period and genre (Source : Project Gutenberg). For convenience the title was kept
in the dataset in order to section for individual models. 

# Dataset Creation Steps
- Removed non-alphabetic characters
- Punctuation (excluding : . ? ! : ')
- Lemmatized words
- removed stop words
- removed words not found in `spacy` english dictionary
- removed sentences with <2 words
- Manual Cleanup


# Findings

After training up a multitude of models we found the models trained on a single short story or book tended to work
the best overall. When analyzing the full corpus the output cosine similarities tended to be under 0.5 which is not
great in comparison to the individual stories which often had output cosine similarites around 0.7.
