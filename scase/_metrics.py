import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from ._utils import *


class Metrics:
    @staticmethod
    def knn_score(embeddings: np.ndarray, labels: np.ndarray, n_neighbors=10) -> int:
        """performs KNN classifier on the spectral-embedding space and computes its accuracy.

        Parameters
        ----------
        embeddings : np.ndarray
            The spectral-embedding space.
        labels : np.ndarray
            The Spectral-embedding points' labels
        n_neighbors : int (default = 5)
            the number of neighbors for computing the KNN algorithm.

        Returns
        -------
        int
            the knn classifier accuracy for the given data.
        """

        knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors + 1, weights=knn_weights_without_self)
        knn_classifier.fit(embeddings, labels)
        knn_pred = knn_classifier.predict(embeddings)
        knn_acc = np.mean((knn_pred == labels) * 1)
        return knn_acc
