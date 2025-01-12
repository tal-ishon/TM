import numpy as np
# Calculate eigenvalues and eigenvectors not with numpy
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix


# Create Gaussian Kernel over the affinity matrix

def _gaussian_kernel(matrix, sigma=1):
    """
    Create a Gaussian kernel over the affinity matrix.
    """
    # Calculate the Gaussian kernel
    kernel = np.exp(-matrix / (2 * sigma**2))

    return kernel


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
    Create the random walk Laplacian matrix from the affinity matrix.
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


def _adjust_eigenvalues_and_vectors(matrix, k=100):
    """
    Adjust the eigenvalues and eigenvectors of the Laplacian matrix.
    
    """
    # Step 1: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigs(matrix)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    
    # Sort eigenvalues and eigenvectors in descending order of eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Step 2: Normalize the most significant eigenvalue to 1
    max_eigenvalue = eigenvalues[0]
    adjusted_eigenvalues = eigenvalues / max_eigenvalue
    
    # Step 3: Select top k eigenvalues and eigenvectors
    top_k_eigenvalues = adjusted_eigenvalues[:k]
    top_k_eigenvectors = eigenvectors[:, :k]
    
    # Step 4: Scale eigenvectors by their respective eigenvalues
    scaled_eigenvectors = top_k_eigenvectors * top_k_eigenvalues
    
    return top_k_eigenvalues, scaled_eigenvectors


def get_random_walk_laplacian(matrix, k=100):
    """
    Create the random walk Laplacian matrix from the affinity matrix.
    """
    matrix = matrix.toarray() # matrix is sparse matrix
    kernel = _gaussian_kernel(matrix)
    random_walk_laplacian = _random_walk_laplacian(kernel)
    top_k_eigenvalues, scaled_eigenvectors = _adjust_eigenvalues_and_vectors(random_walk_laplacian, k=k)

    return top_k_eigenvalues, scaled_eigenvectors


def get_gaussian_kernel(matrix, sigma=1, k=100):
    """
    Create the Gaussian kernel over the affinity matrix.
    """
    matrix = matrix.toarray()
    return _gaussian_kernel(matrix, sigma=sigma)

