import numpy
import torch
from torch.nn import functional as F
from WordEmbedding import SkipGram, Word2Vec

def get_similar_words(model, w2v, word, top_n=10):
    if word not in w2v.vocab:
        print(f"'{word}' not in vocabulary.")
        return []

    # Get the index of the word
    word_idx = w2v.vocab[word]

    # Extract the input embedding weights
    embeddings = model.input_embeddings.weight.data

    # Normalize the embeddings
    norm_embeds = F.normalize(embeddings, p=2, dim=1)

    # Get the embedding vector for the input word
    target_embed = norm_embeds[word_idx].unsqueeze(0)  # shape: (1, embedding_dim)

    # Compute cosine similarities
    similarities = torch.matmul(norm_embeds, target_embed.T).squeeze()  # shape: (V,)

    # Get top N similar indices (excluding the word itself)
    sim_scores, sim_indices = torch.topk(similarities, top_n + 1)
    sim_indices = sim_indices.tolist()
    sim_scores = sim_scores.tolist()

    # Remove the input word itself from the results
    similar_words = [(w2v.callback[idx], sim_scores[i]) for i, idx in enumerate(sim_indices) if idx != word_idx][:top_n]

    return similar_words


def word_arithmetic(model, w2v, pos_words, neg_words, top_n=5):
    """
    Computes vector arithmetic: (pos1 + pos2 + ...) - (neg1 + neg2 + ...)
    and returns top_n most similar words.
    """
    embeddings = model.input_embeddings.weight.data
    norm_embeds = F.normalize(embeddings, p=2, dim=1)

    # Get vectors for positive and negative words
    pos_vecs = []
    for word in pos_words:
        if word in w2v.vocab:
            pos_vecs.append(norm_embeds[w2v.vocab[word]])
        else:
            print(f"Warning: '{word}' not in vocabulary.")

    neg_vecs = []
    for word in neg_words:
        if word in w2v.vocab:
            neg_vecs.append(norm_embeds[w2v.vocab[word]])
        else:
            print(f"Warning: '{word}' not in vocabulary.")

    if not pos_vecs:
        print("No valid positive words.")
        return []

    # Compute result vector
    result_vector = torch.stack(pos_vecs).sum(dim=0)
    if neg_vecs:
        result_vector -= torch.stack(neg_vecs).sum(dim=0)

    # Normalize result vector
    result_vector = F.normalize(result_vector.unsqueeze(0), p=2, dim=1)

    # Compute cosine similarity to all words
    similarities = torch.matmul(norm_embeds, result_vector.T).squeeze()

    # Get top N matches
    sim_scores, sim_indices = torch.topk(similarities, top_n + len(pos_words) + len(neg_words))
    sim_indices = sim_indices.tolist()
    sim_scores = sim_scores.tolist()

    # Filter out input words from result
    input_words = set(pos_words + neg_words)
    result = []
    for i, idx in enumerate(sim_indices):
        word = w2v.callback[idx]
        if word not in input_words:
            result.append((word, sim_scores[i]))
        if len(result) == top_n:
            break

    return result


# Assume `model` and `w2v` are trained
w2v = Word2Vec("Sample.csv", 100, 3)
model = SkipGram(w2v)
state = torch.load("model.pt")
model.load_state_dict(state)
word = 'man'
#
# result = word_arithmetic(model, w2v, pos_words=["count", "vampire"], neg_words=[], top_n=5)
#
# print("Result of 'queen - girl + boy':")
# for word, score in result:
#     print(f"{word}: {score:.4f}")

similar = get_similar_words(model, w2v, word, top_n=10)
print(f"Top related words to {word}:")
for word, score in similar:
    print(f"{word}: {score:.4f}")
