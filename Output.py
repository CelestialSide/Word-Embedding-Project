import numpy
import torch
from torch.nn import functional as F
from WordEmbedding import SkipGram, Word2Vec

def get_similar_words(model, w2v, word, n = 10):
    if word not in w2v.vocab:
        print(f"'{word}' not in vocabulary.")
        return []

        # Get the index of the word
    word_index = w2v.vocab[word]

        # Extract the input embedding weights
    embeddings = model.input_embeddings.weight.data

        # Normalize the embeddings
    norm_embeds = F.normalize(embeddings, p = 2, dim = 1)

        # Get the embedding vector for the input word
    target_embed = norm_embeds[word_index].unsqueeze(0)  # shape: (1, embedding_dim)

        # Compute cosine similarities and find n best results
    similarities = torch.matmul(norm_embeds, target_embed.T).squeeze()  # shape: (V,V)

        # n + 1 removes target word as it will always be best result
    scores, indices = torch.topk(similarities, n + 1)
    indices = indices.tolist()
    scores = scores.tolist()

        # Remove the input word itself from the results
    similar_words = [(w2v.callback[index], scores[i]) for i, index in enumerate(indices) if idx != word_index][:n]

    return similar_words


    # When finding Similarities make sure to use the dataset that matches the trained model
w2v = Word2Vec("DataSet/Cleaned_Corpus.csv", 100, 5)

model = SkipGram(w2v)
state = torch.load("Models/full_model.pt")
model.load_state_dict(state)
word = 'monster'


similar = get_similar_words(model, w2v, word, n = 3)
print(f"Top related words to {word}:")
for word, score in similar:
    print(f"{word}: {score:.4f}")
