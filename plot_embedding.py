import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import torch
import umap
from scase._reduction import ScaSE
from scase._utils import get_affinity_matrix, get_eigenvectors
from scase._cluster import SpectralNet
# from spectralnet._reduction import SpectralReduction
from sklearn.manifold import SpectralEmbedding
from scipy.sparse import csr_matrix


def perform_pca():
    # Perform PCA
    pca = PCA(n_components=2)
    word_embeddings_2d = pca.fit_transform(word_embeddings)
    print("After PCA")
    return word_embeddings_2d


def perform_UMAP():
    reducer = umap.UMAP()
    embed = reducer.fit_transform(word_embeddings)
    return embed


def first_2_eigenvec():
    SE = SpectralEmbedding(n_components=50)
    eigenvec = SE.fit_transform(word_embeddings)

    return eigenvec



HOME = "/home/tal/dev/TopicModeling/Results/CleanerCorpus/TF-IDF"
word_embeddings = torch.load(f"{HOME}/Affinity_matrix")
# word_embeddings = torch.tensor(word_embeddings, dtype=torch.float32) # already a tensor type
word_embeddings = csr_matrix(word_embeddings)
words_dict = torch.load(f"{HOME}/word_to_ix")

words = [key for key in words_dict.keys()]
TYPE = "UMAP_AffinityMat"

# word_embeddings_2d = perform_pca()
word_embeddings_2d = perform_UMAP(first_2_eigenvec())
word_list = ["dog", "cat", "pet", "computer", "man", "woman", "king", "queen", "apple", "banana", "game", "player",
             "ball", "horse", "television", "politic", "arab", "israeli", "america"]


# Plot the words in 2D using PCA
plt.figure(figsize=(10, 8))
for i, word in enumerate(words):
    if word not in word_list:
        continue
    plt.scatter(word_embeddings_2d[i, 0], word_embeddings_2d[i, 1])
    plt.text(word_embeddings_2d[i, 0] + 0.01, word_embeddings_2d[i, 1] + 0.01, word, fontsize=12)

plt.title(f"Word Embeddings plotted using {TYPE}")
plt.xlabel(f"{TYPE} Component 1")
plt.ylabel(f"{TYPE} Component 2")
plt.grid(True)
plt.savefig(f"{TYPE}_TFIDF")