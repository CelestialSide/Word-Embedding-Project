# Overview
This project intends to create a Word2Vec model from scratch in order to find semantic relationships
between words in 1800s Gothic Literature.

# Contents
- Models
  - All models that were trained, associated dataset is found in folder for output
- Dataset
  - Story Database and File Creation were old attempts and were not used to create final dataset
  - Corpus Cleaning created our final dataset(s), full dataset is found in model under `Full Model`
- Output.py : This lets you analyze different relationships
  - When using Output.py please comment out the train line in WordEmbedding.py or it will train a new model
  - You may also comment out the following code in the Word2Vec class to decrease time as it is not required for output
    ```
    neg_context = self.create_negative_context()
    self.neg_context = neg_context
    ```
- WordEmbedding.py : Trains model on dataset given (Full dataset took about 15 hours to train)

# Dataset
We created a dataset from scratch for this project. In order to create the dataset we first pulled the raw text
of many books within the time period and genre (Source : Project Gutenberg). The title was kept in the dataset
for reference so that we could section off specific stories for individual models. From there the dataset was 
cleaned in order to make the dataset compatible with the Embedding model and to reduce the dictionary in order
to speed up training time.


# Dataset Creation Steps
- Removed non-alphabetic characters.
- Punctuation (excluding : . ? ! : ').
- Lemmatized words.
- Removed stop words.
- Removed words not found in `spacy` English dictionary.
- Removed sentences with <2 words.
- Manual Cleanup.

# Preprocessing
Before training, some preprocessing steps needed to be done. For the full model, we chose a window size of 5, which created
positive pairs dependent on what surrounded the word. These positive pairs were then stored as a frequency list. For every positive pair, 5 negative pairs were sampled and stored as a frequency list. Finally the target word, positive or negative sample, and label were stored in a torch dataset for dataloading.

# Training
- Input Layer:
  - One-hot vector of shape 23,188 x 1.
  - Represents the target word in our vocabulary.
- Embedding Layer:
  - Input is multiplied by Embedding Vector.
    - Size 1 x 100.
    - Embedding Vector is initialized with weights randomized with uniform randomness.
  - This results in the embedding layer of shape 23,188 by 100.
- Hidden Layer:
  - Splits into two branches:
    - Top Branch: Positive Samples.
    - Bottom Branch: Negative Samples.
  - Both branches are updated during batching to train in real weights.
    - Resulting Vector's shape is not altered.
    - Multiple pairs (target, context) are processed in parallel.
    - Input Embedding is stacked into 64 matrices of size 23,188 by 100.
  - Increases training efficiency and gradient stability.
- Output:
  - The top branch is multiplied by the transpose of the bottom branch (Matrix Multiplication).
  - Result is a 23,188 x 23,188 Matrix which contains cosine similarites.


# Findings

After training up a multitude of models, we found the models trained on a single short story or book tended to work the best overall. When analyzing the full corpus, the output cosine similarities tended to be at or around 0.5, which is not great in comparison to the individual stories which often had output cosine similarities around 0.7. We believe the reason for this discrepancy is that authors tend to write in different styles that often place common words, such as `man`, in with different context pairings, which causes the overall similarities to be low. Rare words (For example `seaman` and `fatal`) tended to output more accurate words, but the score was still low. In single story datasets names, in particular, tended to result in higher scores.
