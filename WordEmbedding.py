import numpy as np
import pandas as pd
import random
import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

class Word2Vec:
        # Prints out some useful information
    def info(self):
        pairs = 0
        for x in self.pos_context.items():
            pairs += len(x[1][0])

        print("Number of Unique Words:", self.V)
        print("Embedding Size:", self.embedding_size)
        print("Window Size:", self.window_size)
        print("Number of paired words:", pairs)

    def __init__(self, file, embed_size = 100, window_size = 3):
        self.word_freq = np.array
        self.window_size = window_size
        self.embedding_size = embed_size

        V, vocab, callback, pos_context, word_freq = self.Preprocessing(file)
        self.V = V
        self.vocab = vocab
        self.callback = callback
        self.pos_context = pos_context

        neg_context = self.create_negative_context()
        self.neg_context = neg_context

        # Preprocessing creates vocab and target word context word pairings
            # Along with some information attached to the two objects
    def Preprocessing(self, file):
        df = pd.read_csv(file)
        corpus = df['Story'].to_numpy()
        for i in np.ndindex(corpus.shape[0]):
            tokens = corpus[i].replace('.', '').split(' ')
            corpus[i] = tokens

        vocab = {}
        index = 0

        for i in np.ndindex(corpus.shape[0]):
            sentence = set(corpus[i])
            for word in sentence:
                if word not in vocab:
                    vocab[word] = index
                    index += 1
        callback = {i: word for word, i in vocab.items()}
        V = len(vocab) # Length of vocab list
        pos_context = {key: [[], 0, np.zeros(V)] for key in callback}

            # Appends Dictionary of Target_Word
                # Context words, Total Number of Pairs, Context Word Frequencies
        for sent in np.ndindex(corpus.shape[0]):
            sentence = corpus[sent]
            for i in range(len(sentence)):
                center = vocab[sentence[i]]
                for j in range(-self.window_size, self.window_size + 1):
                    if j != 0 and 0 <= (i + j) < len(sentence):
                        context = vocab[sentence[i + j]]
                        pos_context[center][2][context] += 1
                        pos_context[center][1] += 1
                        if context not in pos_context[center][0]:
                            pos_context[center][0].append(context)
        # Total context frequency (for all words)
        word_freq = np.zeros(V, dtype=float)
        for key in pos_context:
            word_freq += pos_context[key][2]  # Sum over all context counts
        self.word_freq = word_freq
        return V, vocab, callback, pos_context, word_freq


    def create_negative_context(self, k = 3):
        # Apply smoothing: freq^0.75
        freq_dist = self.word_freq ** 0.75
        freq_dist /= freq_dist.sum()  # Normalize to make it a probability distribution

        # Pre-sample a large number of negative samples to speed up training
        neg_sample_pool = np.random.choice(
            np.arange(self.V), size=1000000, p=freq_dist)
        pool_index = 0

        neg_context = {key: np.zeros(self.V, dtype=int) for key in self.pos_context}

        for key in tqdm.tqdm(neg_context):
            for _ in range(self.pos_context[key][1]):
                tracker = 0
                while tracker < k:
                    neg_sample = neg_sample_pool[pool_index]
                    pool_index = (pool_index + 1) % len(neg_sample_pool)
                    if neg_sample != key and neg_sample not in self.pos_context[key][0]:
                        neg_context[key][neg_sample] += 1
                        tracker += 1
        return neg_context

    # Initializes Data in tuple (Target_Word, Positive/Negative_Sample, 0/1)
class GothicDataset(Dataset):
    def __init__(self, pos_samples, neg_samples):
        pos_context = {key: pos_samples[key][2] for key in pos_samples}

        self.data = []
        for target in pos_context:
            pos_counts = pos_context[target]
            neg_counts = neg_samples[target]

            # Positive samples
            pos_indices = np.where(pos_counts > 0)[0]
            for index in pos_indices:
                count = int(pos_counts[index])
                self.data.extend([(target, index, 1)] * count)

            # Negative samples
            neg_indices = np.where(neg_counts > 0)[0]
            for index in neg_indices:
                count = neg_counts[index]
                self.data.extend([(target, index, 0)] * count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SkipGram(nn.Module):
    def __init__(self, w2v):
        super(SkipGram, self).__init__()
        self.V = w2v.V
        self.embedding_size = w2v.embedding_size

            # Target word embeddings (INPUT)
        self.input_embeddings = nn.Embedding(self.V, self.embedding_size)
            # Context word embeddings (OUTPUT)
        self.output_embeddings = nn.Embedding(self.V, self.embedding_size)

            # Initialize weights
        self.embedding_weight()

    def embedding_weight(self):
        nn.init.xavier_uniform_(self.input_embeddings.weight)
        nn.init.xavier_uniform_(self.output_embeddings.weight)

    def forward(self, target, context, labels):
            # shape: (batch_size, embedding_size)
        target_embeds = self.input_embeddings(target)
        context_embeds = self.output_embeddings(context)

            # Dot product between target and context embeddings
        dot_product = torch.sum(target_embeds * context_embeds, dim=1)

            # Get Probabilities of context embeddings
        log_probs = torch.sigmoid(dot_product)

            # Calculate Loss
        loss = F.binary_cross_entropy(log_probs, labels.float())

        return loss


def train(corpus, num_epochs = 100, batch_size = 64, lr = 0.001):
        # Preprocessing
    w2v = Word2Vec(corpus, 100, 5)
    w2v.info()

        # Initialize Dataset
    gd = GothicDataset(w2v.pos_context, w2v.neg_context)
    dataloader = DataLoader(gd, batch_size=batch_size, shuffle=True)

        # Initialize model
    model = SkipGram(w2v)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in (range(num_epochs)):
        running_loss = 0
        for _, data in tqdm.tqdm(enumerate(dataloader, 0)):
            target, context, label = data

            optimizer.zero_grad()
            loss = model(target, context, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1} of {num_epochs}, Loss: {running_loss / len(gd):.10f}")

    w2v.info()
    torch.save(model.state_dict(), "Models/Cask_Model.pt")
    return model

train("Cleaned_Corpus.csv", 30)




