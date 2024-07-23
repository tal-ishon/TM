import numpy as np
from Embeddings import word2vec


def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def most_similar(word, k=5):
    if word not in word2vec:
        return f"{word} not found in embeddings."

    word_vector = word2vec[word]
    similarities = {}  # dictionary to store part2.pdf. key: word, value: similarity to "word"

    for w, v in word2vec.items():
        if w != word:
            similarities[w] = cosine_similarity(word_vector, v)

    sorted_similarities = sorted(
        similarities.items(), key=lambda item: item[1], reverse=True)  # sort by similarity in descending order
    return sorted_similarities[:k]  # return top k most similar words


# Example usage:
words_to_check = ['dog', 'england', 'john', 'explode', 'office', 'king', 'computer', 'apple']

for word in words_to_check:
    similar_words = most_similar(word)
    print(f"The most similar words to '{word}' are:")
    for similar_word, similarity in similar_words:
        print(f"{similar_word}: {similarity:.4f}")
    print()