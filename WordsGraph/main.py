import graph
from gensim import corpora
import pickle
from scipy.sparse import save_npz
import torch


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


def save_matrix_torch(matrix, soft_mat=False, save_path="affinity_matrix.pt"):

    if soft_mat:
        affinity_dense = matrix.toarray()

    # Convert to PyTorch tensor
    affinity_tensor = torch.tensor(affinity_dense, dtype=torch.float32)

    # Save the PyTorch tensor to a file
    torch.save(affinity_tensor, save_path)


def main():
    path = "/home/dsi/ishonta/TM/ProcessedData/20NewsGroup"
    dictionary, _, _, corpus = load_data(path)

    words_graph = graph.Graph(dictionary=dictionary, corpus=corpus)
    affinity_matrix = words_graph.get_graph_as_affinity_matrix()
    matrix = soft_matrix(matrix=affinity_matrix)
    save_matrix_torch(matrix, True)


if __name__ == "__main__":
    main()