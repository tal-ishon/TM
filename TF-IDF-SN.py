from tfidf_utils import generate_tfidf_matrix, prepare_data, get_filtered_corpus
from scase._utils import get_affinity_matrix, get_eigenvectors
from scase._cluster import SpectralNet
import torch
import os
import numpy as np
from preprocessing import prepare_cleaner_data
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import NearestNeighbors


def affinity_matrix(X, n_neighbors=10):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    affinity_matrix = np.zeros((X.shape[0], X.shape[0]))
    
    for i in range(X.shape[0]):
        for j in range(1, n_neighbors):
            affinity_matrix[i, indices[i][j]] = np.exp(-distances[i][j] ** 2)
            affinity_matrix[indices[i][j], i] = affinity_matrix[i, indices[i][j]]  # ensure symmetry
    
    return torch.tensor(affinity_matrix, dtype=torch.float32)


def get_corpus():
    data = prepare_cleaner_data()
    word_to_ix, corpus = get_filtered_corpus(data)
    return word_to_ix, corpus


def create_laplacian_graph():
    word_to_ix, corpus = get_corpus()
    torch.save(word_to_ix, f"{HOME}/word_to_ix")
    exit(0)
    tf_idf_mat, feature_names = generate_tfidf_matrix(corpus)
    
    W = affinity_matrix(tf_idf_mat.transpose())
    
    torch.save(W, f"{HOME}/Affinity_matrix")
    print("got W")
    return W


def save_pred(pred):
    torch.save(pred, name)


def SpectralNet_fit_and_predict(W):
    SN = SpectralNet(n_clusters=TOP)
    print("start fit")
    SN.fit(W)

    print("start predict")
    predictions = SN.predict(W)

    save_pred(predictions)

    
def main(no_graph=True):
    if no_graph:
        # if don't have graph already
        W = create_laplacian_graph()
    else:
        # if have a graph
        W = torch.load(f"{HOME}/Affinity_matrix")
    # eigenvectors = get_eigenvectors(W)
    # top_eigenvec = eigenvectors[:TOP]
    SpectralNet_fit_and_predict(W)


TOP = 104  # according to number of topics
HOME = os.getcwd() + "/Results/CleanerCorpus/TF-IDF"
name = f"{HOME}/{TOP}_topics_pred_cleaner_tfidf"
# main(True)
create_laplacian_graph()