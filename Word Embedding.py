import numpy as np
import pandas as pd
import random
import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

class Word2Vec:
    def info(self):
        pairs = 0
        for x in self.pos_context.items():
            pairs += len(x[1][0])

        print("Number of Unique Words:", self.V)
        print("Embedding Size:", self.embedding_size)
        print("Window Size:", self.window_size)
        print("Number of paired words:", pairs)

    def __init__(self, file, embed_size = 100, window_size = 3):
        self.window_size = window_size
        self.embedding_size = embed_size

        V, vocab, callback, pos_context = self.Preprocessing(file)
        self.V = V
        self.vocab = vocab
        self.callback = callback
        self.pos_context = pos_context

        neg_context = self.create_negative_context(5)
        self.neg_context = neg_context


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
        V = len(vocab)
        pos_context = {key: [[], 0, np.zeros(V)] for key in callback}
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
        return V, vocab, callback, pos_context

        # Returns a list of numbers Ex. [0, 1, 0, 1, 3]
            # (5 words exist in vocab)
    def create_negative_context(self, k = 5):
        neg_context = {key: np.zeros(self.V, dtype=int) for key in self.pos_context}
        tracker = 0

        for key in tqdm.tqdm(neg_context):
            for i in range(self.pos_context[key][1]):
                tracker = 0
                while tracker < k:
                    neg_sample = random.randint(0, self.V - 1)
                    if neg_sample not in self.pos_context[key][0] and neg_sample != key:
                        neg_context[key][neg_sample] += 1
                        tracker += 1
        return neg_context


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

        # Target (center) word embeddings
        self.input_embeddings = nn.Embedding(self.V, self.embedding_size)
        # Context (output) word embeddings
        self.output_embeddings = nn.Embedding(self.V, self.embedding_size)

        # Initialize weights
        self.embedding_weight()

    def embedding_weight(self):
        nn.init.xavier_uniform_(self.input_embeddings.weight)
        nn.init.xavier_uniform_(self.output_embeddings.weight)

    def forward(self, target_idxs, context_idxs, labels):
        # shape: (batch_size, embedding_size)
        target_embeds = self.input_embeddings(target_idxs)
        context_embeds = self.output_embeddings(context_idxs)

        # Dot product between target and context embeddings
        dot_product = torch.sum(target_embeds * context_embeds, dim=1)

        # Get Probabilities of context embeddings
        log_probs = torch.sigmoid(dot_product)

        # Calculate Loss
        loss = F.binary_cross_entropy(log_probs, labels.float())

        return loss


def train(corpus, num_epochs=10, batch_size = 64, lr = 0.001):
        # Preprocessing
    w2v = Word2Vec(corpus, 100, 3)
    w2v.info()

        # Initialize Dataset
    gd = GothicDataset(w2v.pos_context, w2v.neg_context)
    dataloader = DataLoader(gd, batch_size=batch_size, shuffle=True)

        # Initialize model
    model = SkipGram(w2v)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

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

        print(f"Epoch {epoch+1} of {num_epochs}, Loss: {running_loss / len(gd):.4f}")

    w2v.info()
    torch.save(model.state_dict(), "DataSet/skip-gram_model.pt")
    return model


# Epoch 10 of 10, Loss: 0.0108
# Number of Unique Words: 23188
# Embedding Size: 100
# Window Size: 3
# Number of paired words: 1574566

train("Cleaned_Corpus.csv", 10)



