import graph
from gensim import corpora
from scipy.sparse import save_npz, csgraph
import torch
from sklearn.mixture import GaussianMixture
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
import numpy as np
from  matrix_utils import get_gaussian_kernel, get_random_walk_laplacian


def load_data(path_data):
    # Load the dictionary from a file
    dictionary = corpora.Dictionary.load(f'{path_data}/lda_dictionary.gensim')

    # Load the BoW corpus from a Matrix Market format file
    bow_corpus = corpora.MmCorpus(f'{path_data}/lda_corpus.mm')
    doc_term_matrix = [list(doc) for doc in bow_corpus]

    texts = [
        list((dictionary[word_id] for word_id, freq in bow))
        for bow in doc_term_matrix
    ]
    
    return dictionary, bow_corpus, doc_term_matrix, texts

def soft_predictions(predictions):
        # Define epsilon
        epsilon = 0.01
        soft_predictions = predictions + epsilon  # Add epsilon to each element
        soft_predictions /= soft_predictions.sum(axis=1, keepdims=True) # Normalize each row so that the sum is 1

        return soft_predictions


def soft_matrix(matrix, epsilon=0.001):
    matrix = matrix.toarray()  # Convert to dense if needed
    smoothed_matrix = matrix.copy()
    
    # Apply epsilon to all zero entries
    smoothed_matrix[smoothed_matrix == 0] = epsilon
    
    return smoothed_matrix


def save_matrix(matrix, save_path="affinity_matrix.pt"):
    # Convert lil_matrix to csr_matrix
    csr_affinity_matrix = matrix.tocsr()

    # Save as .npz file
    save_npz("affinity_matrix.npz", csr_affinity_matrix)


def save_matrix_torch(matrix, soft_mat=False, save_path="affinity_matrix_npmi.pt"):
    """
    Save the affinity matrix as a PyTorch tensor.
    Supports optional smoothing.
    """
    if soft_mat:
        # Apply smoothing
        matrix = soft_matrix(matrix)

    # Convert the matrix to a dense NumPy array (if it is sparse)
    if not isinstance(matrix, np.ndarray):
        matrix = matrix.toarray()

    # Convert to PyTorch tensor with dtype float32
    affinity_tensor = torch.tensor(matrix, dtype=torch.float32)

    # Save the dense tensor
    torch.save(affinity_tensor, save_path)

    print(f"Matrix saved to {save_path} as a dense PyTorch tensor.")


def _apply_gmm(word_features, path, n_components=20, to_save=True, metric="npmi"):
    """
    Apply Gaussian Mixture Model (GMM) to the word features.
    """
    from sklearn.decomposition import PCA
    pca_components = 50
    # Apply PCA for dimensionality reduction
    if pca_components and pca_components < word_features.shape[1]:
        pca = PCA(n_components=pca_components)
        word_features = pca.fit_transform(word_features)

    gmm = GaussianMixture(n_components=n_components, 
                          n_init=5,
                          verbose=1, 
                          random_state=42)
    gmm.fit(word_features)

    topic_assignments = gmm.predict_proba(word_features)
    smoothed_topic_assignments = soft_predictions(topic_assignments)
    if to_save:
        save_matrix_torch(smoothed_topic_assignments, save_path=f"{path}/_{metric}_topic_assignments.pt")
    else:
        return smoothed_topic_assignments


def create_prior(path, word_features, n_components, metric="npmi", to_save=True):
    """
    Create a prior from the affinity matrix.
    """
    _apply_gmm(word_features, path, n_components=n_components, to_save=to_save, metric=metric)     # Pass the features to the GMM application function



def main(metrics=["pmi"]):
    path = "ProcessedData/20NewsGroup"
    save_prior_path = "WordsGraph/priors/RandomWalk/Similarity/Second"

    dictionary, _, _, corpus = load_data(path)

    for m in metrics:
        words_graph = graph.Graph(dictionary=dictionary, corpus=corpus, metric=m, similarity=True)
        affinity_matrix = words_graph.get_graph_as_affinity_matrix()
        _, matrix = get_random_walk_laplacian(affinity_matrix, k=500)
        create_prior(save_prior_path, matrix, n_components=100, to_save=True, metric=m)


if __name__ == "__main__":
    import sys
    argv = sys.argv

    argc = len(argv)    
    if argc > 1:
        metrics = []
        for v in argv[1:]:
            metrics.append(v)
        main(metrics)
    else:
        main()
