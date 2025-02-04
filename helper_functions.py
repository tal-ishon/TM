import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from gensim.models import CoherenceModel

def diffusion_maps(data, n_components=2, k=None, epsilon=None, scaling_factor=10, diffusion_time=1):
    n_samples = data.shape[0]

    # Compute cosine similarity matrix
    distances = cosine_distances(data)

    # Sparsify using k-nearest neighbors if k is specified
    if k is not None:
        neigh = NearestNeighbors(n_neighbors=k+1, metric='cosine')
        neigh.fit(data)
        top_k_indices = neigh.kneighbors(data, return_distance=False)

        # Create a mask for KNN filtering
        mask = np.zeros((n_samples, n_samples), dtype=bool)
        rows = np.arange(n_samples)[:, None]
        mask[rows, top_k_indices] = True

        # Set all non-KNN connections in distances to infinity
        distances[~mask] = np.inf

    # Set the diagonal (self-distances) to infinity for subtraction step
    np.fill_diagonal(distances, np.inf)
    min_distances = np.min(distances, axis=1)
    distances -= min_distances[:, np.newaxis]
    np.fill_diagonal(distances, 0)

     # Dynamically calculate epsilon (sigma) as the median pairwise distance if not provided
    if epsilon is None:
        finite_distances = distances[np.isfinite(distances)]  # Ignore infinities
        epsilon = scaling_factor * np.median(finite_distances)
        print(f"Computed epsilon (median pairwise distance): {epsilon}")


    # Compute the Gaussian kernel with sigma controlling the bandwidth of the kernel
    W = np.exp(-distances ** 2 / (2 * epsilon ** 2))

    # Normalize to construct the Markov matrix
    W_sparse = csr_matrix(W)
    row_sums = W_sparse.sum(axis=1).A1
    D_inv = csr_matrix(np.diag(1.0 / row_sums))
    P = D_inv @ W_sparse

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigs(P, k=n_components + 1, which='LR')
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    # Skip the first eigenvector (trivial constant one)
    diffusion_coords = eigenvectors[:, 1:] * eigenvalues[1:] ** diffusion_time

    return diffusion_coords

# C_V Coherence calculations

# Calculate coherence metrics for all models
def coherence_model_evaluation(topics, texts, dictionary, coherence="c_v"):  
    coherence_model = CoherenceModel(
        topics=topics, 
        texts=texts, 
        dictionary=dictionary, 
        coherence=coherence
    )
    return coherence_model.get_coherence()

def topic_diversity_evaluation(topics, topk=10):
    """
    compute the proportion of unique words

    Parameters
    ----------
    topics: a list of lists of words
    topk: top k words on which the topic diversity will be computed
    """
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than '+str(topk))
    else:
        unique_words = set()
        for topic in topics:
            unique_words = unique_words.union(set(topic[:topk]))
        puw = len(unique_words) / (topk * len(topics))
        return puw


