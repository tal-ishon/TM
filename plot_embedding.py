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
from utils import load_file_txt


def perform_pca(word_embeddings):
    # Perform PCA
    pca = PCA(n_components=2)
    word_embeddings_2d = pca.fit_transform(word_embeddings)
    print("After PCA")
    return word_embeddings_2d


def perform_UMAP(word_embeddings):
    reducer = umap.UMAP()
    embed = reducer.fit_transform(word_embeddings)
    return embed


def first_2_eigenvec():
    SE = SpectralEmbedding(n_components=50)
    eigenvec = SE.fit_transform(word_embeddings)

    return eigenvec



word_embeddings = torch.load(f"Results/WordEmbedding/embedding")
# word_embeddings = torch.tensor(word_embeddings, dtype=torch.float32) # already a tensor type
word_embeddings = csr_matrix(word_embeddings)
words_dict = torch.load("Results/WordEmbedding/word_to_ix")

words = [key for key in words_dict.keys()]
# words = load_file_txt("Results/Wo/Tal_words")
TYPE = "PCA"

# word_embeddings_2d = perform_pca()
ev_word_embed = first_2_eigenvec()
word_embeddings_2d = perform_pca(ev_word_embed)
word_list = ["dog", "cat", "pet", "computer", "man", "woman", "king", "queen", "apple", "banana", "food", 
             "tomato", "game", "player","ball", "horse", "television", "politic", "arab", "israeli", "america",
             "pc", "intel", "car", "motorcycle", "sport"]

# Scale the data
scaling_factor = 1e5  # Adjust this factor based on the range of your data
word_embeddings_2d = word_embeddings_2d * scaling_factor

# Plot the words in 2D using PCA
plt.figure(figsize=(10, 8))
for i, word in enumerate(words):
    if word not in word_list:
        continue
    print(f"Plotting: {word}")
    plt.scatter(word_embeddings_2d[i, 0], word_embeddings_2d[i, 1])
    plt.text(word_embeddings_2d[i, 0] + 0.05, word_embeddings_2d[i, 1] + 0.05, word, fontsize=12)


plt.title(f"Word Embeddings plotted using {TYPE}")
plt.xlabel(f"{TYPE} Component 1")
plt.ylabel(f"{TYPE} Component 2")
plt.tight_layout()  # Ensure plot elements are not cut off
plt.grid(True)
plt.savefig(f"{TYPE}_SE_WE")