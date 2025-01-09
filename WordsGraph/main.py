import graph
from gensim import corpora
import pickle
from scipy.sparse import save_npz
import torch
import numpy as np


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


def main():
    path = "/home/dsi/ishonta/TM/ProcessedData/20NewsGroup"
    dictionary, _, _, corpus = load_data(path)

    words_graph = graph.Graph(dictionary=dictionary, corpus=corpus, metric="npmi")
    affinity_matrix = words_graph.get_graph_as_affinity_matrix()
    save_matrix_torch(affinity_matrix, True)


if __name__ == "__main__":
    main()