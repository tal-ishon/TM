import numpy as np
# Calculate eigenvalues and eigenvectors not with numpy
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix, diags
from gensim import corpora
from scipy.sparse import save_npz, csgraph
import torch
from sklearn.mixture import GaussianMixture
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
import os
from scipy.linalg import fractional_matrix_power


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

def shift_embeddings_to_positive(embeddings):
    """
    Shift the diffusion map embeddings to ensure all values are positive.

    Parameters:
        embeddings (numpy.ndarray): The input embeddings (shape: n_samples x n_dimensions).

    Returns:
        numpy.ndarray: The shifted embeddings with all positive values.
    """
    # Find the minimum value in the entire embedding matrix
    min_val = np.min(embeddings)
    
    # If the minimum value is negative, shift all values by its absolute magnitude
    if min_val < 0:
        shifted_embeddings = embeddings + abs(min_val)
    else:
        # If already positive, no need to shift
        shifted_embeddings = embeddings
    
    return shifted_embeddings

def softmax_with_temperature(matrix, temperature=1.0):
    """
    Compute the softmax for each row of a matrix with temperature scaling.

    Parameters:
        matrix (numpy.ndarray): Input matrix (n_rows x n_columns).
        temperature (float): Temperature parameter. Lower values make it closer to one-hot.

    Returns:
        numpy.ndarray: Row-wise softmax with temperature scaling.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    
    # Stabilize by subtracting the max value for numerical stability
    row_max = np.max(matrix, axis=1, keepdims=True)
    stabilized_matrix = matrix - row_max
    
    # Apply temperature scaling
    scaled_matrix = stabilized_matrix / temperature
    
    # Compute the exponentials and normalize
    exp_matrix = np.exp(scaled_matrix)
    row_sum = np.sum(exp_matrix, axis=1, keepdims=True)
    softmax_matrix = exp_matrix / row_sum
    
    return softmax_matrix


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
    gmm = GaussianMixture(n_components=n_components, 
                          n_init=5,
                          verbose=1, 
                          random_state=42)
    gmm.fit(word_features)

    topic_assignments = gmm.predict_proba(word_features)
    smoothed_topic_assignments = soft_predictions(topic_assignments)
    # smoothed_topic_assignments = softmax_with_temperature(topic_assignments, 1)
    if to_save:
        # Create directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
        save_matrix_torch(smoothed_topic_assignments, save_path=f"{path}/{metric}_topic_assignments.pt")
    else:
        return smoothed_topic_assignments


def create_prior(path, word_features, n_components, metric="npmi", to_save=True, whose_idea="tal"):
    """
    Create a prior from the affinity matrix.
    """
    # if whose_idea == "tal":
    #     word_features = _apply_pca(word_features=word_features, pca_components=50)

    _apply_gmm(word_features, path, n_components=n_components, to_save=to_save, metric=metric)     # Pass the features to the GMM application function


def _apply_pca(word_features, pca_components=50):
    from sklearn.decomposition import PCA

    # Apply PCA for dimensionality reduction
    if pca_components and pca_components < word_features.shape[1]:
        pca = PCA(n_components=pca_components)
        word_features = pca.fit_transform(word_features)
    
    return word_features

# Create Gaussian Kernel over the affinity matrix

def _gaussian_kernel(matrix, sigma=1):
    """
    Create a Gaussian kernel over the affinity matrix.
    """
    # Calculate the Gaussian kernel
    kernel = np.exp(-matrix / (2 * sigma**2))

    return kernel


def _get_more_similar(matrix):
    return np.exp(matrix)


def _laplacian(matrix):
    """
    Create the Laplacian matrix from the affinity matrix.
    """
    # Calculate the degree matrix
    degree_matrix = np.diag(matrix.sum(axis=1))

    # Calculate the Laplacian matrix
    laplacian = degree_matrix - matrix

    return laplacian


def _normalized_laplacian(matrix):
    """
    Create the normalized Laplacian matrix from the affinity matrix.
    """
    # Calculate the degree matrix
    degree_matrix = np.diag(matrix.sum(axis=1))

    # Calculate the Laplacian matrix
    laplacian = degree_matrix - matrix

    # Calculate the normalized Laplacian matrix
    normalized_laplacian = np.dot(np.linalg.inv(degree_matrix), laplacian)

    return normalized_laplacian


def _random_walk_laplacian(matrix):
    """
    Create the random walk Laplacian matrix from the affinity matrix - Tal's idea.
    """
    W = csr_matrix(matrix)
    row_sums = W.sum(axis=1).A1
    D = csr_matrix(np.diag(matrix.sum(axis=1)))
    D_inv = csr_matrix(np.diag(1.0 / row_sums))

    # Calculate the Laplacian matrix
    L = D - W

    # Calculate the random walk Laplacian matrix
    random_walk_laplacian = D_inv @ L

    return random_walk_laplacian



def _nir_random_walk_laplacian(matrix):
    """
    Create the random walk Laplacian matrix from the affinity matrix - Tal's idea.
    
    Parameters:
    - matrix (array-like): Input affinity matrix (dense or sparse).
    
    Returns:
    - random_walk_laplacian (csr_matrix): Random walk Laplacian matrix in sparse form.
    """
    # Ensure the input matrix is in sparse CSR format
    W = csr_matrix(matrix)

    # Compute the degree vector (row sums of W)
    row_sums = np.array(W.sum(axis=1)).flatten()

    # Handle division by zero for isolated nodes (avoid NaNs)
    row_sums[row_sums == 0] = 1.0  # Prevent division by zero

    # Construct sparse degree matrices
    D_inv = diags(1.0 / row_sums)  # D^(-1)

    # Calculate the random walk Laplacian: L_rw = D^(-1) @ (D - W)
    random_walk_laplacian = D_inv @ (diags(row_sums) - W)

    return random_walk_laplacian



def _diffusion_operator(matrix, t=1):
    W = csr_matrix(matrix)
    row_sums = W.sum(axis=1).A1.flatten()
    D_inv = csr_matrix(np.diag(np.where(row_sums != 0, 1.0 / row_sums, 0)))
    
     # Calculate the random walk Laplacian matrix
    P = D_inv @ W

    return P


def _adjust_eigenvalues_and_vectors(matrix, k=100):
    """
    Adjust the eigenvalues and eigenvectors of the Laplacian matrix.
    
    """
    # Step 1: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigs(matrix)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    
    # print(eigenvalues)
    # eigenvalues = 1 - eigenvalues
    # print(eigenvalues)
    
    
    # Sort eigenvalues and eigenvectors in descending order of eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # return eigenvalues, eigenvectors
    
    # Step 2: Normalize the most significant eigenvalue to 1
    max_eigenvalue = eigenvalues[0]
    adjusted_eigenvalues = eigenvalues / max_eigenvalue
    
    print(adjusted_eigenvalues)
    # Step 3: Select top k eigenvalues and eigenvectors
    top_k_eigenvalues = adjusted_eigenvalues[:k]
    top_k_eigenvectors = eigenvectors[:, :k]
    
    # Step 4: Scale eigenvectors by their respective eigenvalues
    scaled_eigenvectors = top_k_eigenvectors * top_k_eigenvalues
    
    return top_k_eigenvalues, scaled_eigenvectors


def get_random_walk_laplacian(matrix, k=100, use_exp=False):
    """
    Create the random walk Laplacian matrix from the affinity matrix.
    """
    matrix = matrix.toarray() # matrix is sparse matrix
    if use_exp:
        matrix = _get_more_similar(matrix)
    random_walk_laplacian = _random_walk_laplacian(matrix)
    top_k_eigenvalues, scaled_eigenvectors = _adjust_eigenvalues_and_vectors(random_walk_laplacian, k=k)

    return top_k_eigenvalues, scaled_eigenvectors


def apply_diffusion_operator(matrix, use_exp=False, k=20):
    matrix = matrix.toarray() # matrix is sparse matrix
    if use_exp:
        matrix = _get_more_similar(matrix)
    do = _diffusion_operator(matrix)
    top_k_eigenvalues, scaled_eigenvectors = _adjust_eigenvalues_and_vectors(do, k=k)

    return top_k_eigenvalues, scaled_eigenvectors


def get_gaussian_kernel(matrix, sigma=1, k=100):
    """
    Create the Gaussian kernel over the affinity matrix.
    """
    matrix = matrix.toarray()
    return _gaussian_kernel(matrix, sigma=sigma)


# Combine features from both Laplacian and Transition Matrix
def combine_features(laplacian_embeddings, transition_embeddings):
    """
    Concatenate features from both Laplacian and Transition Matrix embeddings.
    """
    return np.hstack([laplacian_embeddings, transition_embeddings])

