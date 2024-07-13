import numpy as np
import torch

from ._cluster import SpectralNet
from ._utils import *


class ScaSE:
    def __init__(
            self,
            n_components: int = None,
            is_sparse_graph: bool = False,
            spectral_hiddens: list = [1024, 1024, 512],
            spectral_max_epochs: int = 200,
            spectral_lr: float = 1e-2,
            spectral_lr_decay: float = 0.1,
            spectral_min_lr: float = 1e-6,
            spectral_patience: int = None,
            spectral_batch_size: int = 2048,
            spectral_n_nbg: int = None,
            spectral_scale_k: int = 15,
            spectral_is_local_scale: bool = True,
            should_true_eigenvectors: bool = True,
            t: int = 0,
            ae_hiddens: list = [512, 512, 2048, 10],
            ae_epochs: int = 40,
            ae_lr: float = 1e-3,
            ae_lr_decay: float = 0.1,
            ae_min_lr: float = 1e-7,
            ae_patience: int = 10,
            ae_batch_size: int = 256,
            should_use_ae: bool = False,
    ):
        """SpectralNet is a class for implementing a Deep learning model that performs spectral clustering.
        This model optionally utilizes Autoencoders (AE) for training.

        Parameters
        ----------
        n_components : int (default=None)
            The number of components to keep.

        should_use_ae : bool, optional (default=False)
            Specifies whether to use the Autoencoder (AE) network as part of the training process.

        is_sparse_graph : bool, optional (default=False)
            Specifies whether the graph Laplacian created from the data is sparse.

        ae_hiddens : list, optional (default=[512, 512, 2048, 10])
            The number of hidden units in each layer of the Autoencoder network.

        ae_epochs : int, optional (default=30)
            The number of epochs to train the Autoencoder network.

        ae_lr : float, optional (default=1e-3)
            The learning rate for the Autoencoder network.

        ae_lr_decay : float, optional (default=0.1)
            The learning rate decay factor for the Autoencoder network.

        ae_min_lr : float, optional (default=1e-7)
            The minimum learning rate for the Autoencoder network.

        ae_patience : int, optional (default=10)
            The number of epochs to wait before reducing the learning rate for the Autoencoder network.

        ae_batch_size : int, optional (default=256)
            The batch size used during training of the Autoencoder network.

        spectral_hiddens : list, optional (default=[1024, 1024, 512, 10])
            The number of hidden units in each layer of the Spectral network (not including the output layer).

        spectral_max_epochs : int, optional (default=30)
            The number of epochs to train the Spectral network.

        spectral_lr : float, optional (default=1e-3)
            The learning rate for the Spectral network.

        spectral_lr_decay : float, optional (default=0.1)
            The learning rate decay factor.

        should_true_eigenvectors : bool, optional (default=True)
            Specifies whether to compute the true eigenvectors of the Laplacian of the input data.

        t : int, optional (default=0)
            The diffusion time for the diffusion map algorithm."""

        self.n_components = n_components
        self.is_sparse_graph = is_sparse_graph
        self.spectral_hiddens = spectral_hiddens
        self.spectral_hiddens.append(self.n_components + 1)
        self.spectral_epochs = spectral_max_epochs
        self.spectral_lr = spectral_lr
        self.spectral_lr_decay = spectral_lr_decay
        self.spectral_min_lr = spectral_min_lr
        self.spectral_patience = spectral_patience
        self.spectral_batch_size = spectral_batch_size
        self.spectral_n_nbg = spectral_n_nbg
        if spectral_n_nbg is None:
            self.spectral_n_nbg = max(5, self.spectral_batch_size // 200)
        self.spectral_scale_k = spectral_scale_k
        self.spectral_is_local_scale = spectral_is_local_scale
        self.X_new = None
        self.ortho_matrix = np.eye(n_components+1)
        self.eigenvalues = np.ones(n_components+1)
        self.t = t
        self.columns = None
        if t > 0:
            self.should_true_eigenvectors = True
        else:
            self.should_true_eigenvectors = should_true_eigenvectors

        # AE
        self.ae_hiddens = ae_hiddens
        self.ae_epochs = ae_epochs
        self.ae_lr = ae_lr
        self.ae_lr_decay = ae_lr_decay
        self.ae_min_lr = ae_min_lr
        self.ae_patience = ae_patience
        self.ae_batch_size = ae_batch_size
        self.should_use_ae = should_use_ae

    def fit(self, X: torch.Tensor, y: torch.Tensor = None):
        """Fit the SpectralNet model to the input data.

        Parameters
        ----------
        X : torch.Tensor
            The input data of shape (n_samples, n_features).

        y: torch.Tensor
            The labels of the input data of shape (n_samples,).

        Returns
        -------
        np.ndarray
            The fitted embeddings of shape (n_samples, n_components).
        """
        X = normalize_data(X)

        if self.n_components is None:
            raise ValueError("The number of components must be specified.")

        self.spectral_batch_size = min(self.spectral_batch_size, X.shape[0])

        # Set the spectral patience if it is not set
        if self.spectral_patience is None:
            n_batches = (X.shape[0] // self.spectral_batch_size)
            if n_batches <= 25:
                self.spectral_patience = 10
            else:
                self.spectral_patience = max(1, 250 // n_batches)

        self._spectralnet = SpectralNet(
            n_clusters=self.n_components,
            spectral_hiddens=self.spectral_hiddens,
            spectral_epochs=self.spectral_epochs,
            spectral_lr=self.spectral_lr,
            spectral_lr_decay=self.spectral_lr_decay,
            spectral_min_lr=self.spectral_min_lr,
            spectral_patience=self.spectral_patience,
            spectral_n_nbg=self.spectral_n_nbg,
            spectral_scale_k=self.spectral_scale_k,
            spectral_is_local_scale=self.spectral_is_local_scale,
            spectral_batch_size=self.spectral_batch_size,
            ae_hiddens=self.ae_hiddens,
            ae_epochs=self.ae_epochs,
            ae_lr=self.ae_lr,
            ae_lr_decay=self.ae_lr_decay,
            ae_min_lr=self.ae_min_lr,
            ae_patience=self.ae_patience,
            ae_batch_size=self.ae_batch_size,
            should_use_ae=self.should_use_ae,
        )

        self._spectralnet.fit(X, y)

        if self.should_true_eigenvectors:
            self.compute_ortho_matrix(X)

    def _predict(self, X: torch.Tensor) -> np.ndarray:
        """Predict embeddings for the input data using the fitted SpectralNet model.

        Parameters
        ----------
        X : torch.Tensor
            The input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            The predicted embeddings of shape (n_samples, n_components).
        """
        self._spectralnet.predict(X)
        return self._spectralnet.embeddings_

    def transform(self, X: torch.Tensor) -> np.ndarray:
        """Transform the input data into embeddings using the fitted SpectralNet model.

        Parameters
        ----------
        X : torch.Tensor
            The input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            The transformed embeddings of shape (n_samples, n_components).

        or

        tuple
            The transformed embeddings of shape (n_samples, n_components) and the eigenvalues of the Laplacian of the input data.
        """
        X = normalize_data(X)
        return (self._predict(X) @ self.ortho_matrix @ np.diag((1 - self.eigenvalues) ** self.t))[:, 1:]

    def fit_transform(self, X: torch.Tensor, y: torch.Tensor = None) -> np.ndarray:
        """Fit the SpectralNet model to the input data and transform it into embeddings.

        This is a convenience method that combines the fit and transform steps.

        Parameters
        ----------
        X : torch.Tensor
            The input data of shape (n_samples, n_features).

        y: torch.Tensor
            The labels of the input data of shape (n_samples,).

        Returns
        -------
        np.ndarray
            The fitted and transformed embeddings of shape (n_samples, n_components).
        """
        self.fit(X, y)
        return self.transform(X)

    def get_eigenvalues(self):
        return self.eigenvalues[1:]

    def _get_laplacian_of_small_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Get the Laplacian of a small batch of the input data

        Parameters
        ----------

        batch : torch.Tensor
            A small batch of the input data of shape (batch_size, n_features).

        Returns
        -------
        np.ndarray
            The Laplacian of the small batch of the input data.



        """

        W = get_affinity_matrix(batch, n_nbg=self.spectral_n_nbg, device=self._spectralnet.device)
        L = get_laplacian(W)
        return L

    def compute_ortho_matrix(self, X: torch.Tensor) -> None:
        """Compute the orthogonal matrix for the spectral embeddings.

        Parameters
        ----------
        X : torch.Tensor
            The input data of shape (n_samples, n_features).
        """
        pred = self._predict(X)
        Lambda = self._get_lambda_on_multi_batches(X)

        try:
            ortho_matrix, eigenvalues_pred, _ = np.linalg.svd(Lambda)
            eigenvalues_pred = eigenvalues_pred.real
            self.ortho_matrix = np.array(ortho_matrix.real)
            indices = np.argsort(eigenvalues_pred)
            self.ortho_matrix = np.array(self.ortho_matrix[:, indices])
            self.eigenvalues = eigenvalues_pred[indices]
        except np.linalg.LinAlgError:
            print("Warning: SVD did not converge")
            self.eigenvalues = np.diag(np.sort(Lambda))

    def _get_lambda_on_multi_batches(self, X) -> np.ndarray:
        """Get the mean eigenvalues matrix of the Laplacian of the input data.

        Returns
        -------
        np.ndarray
            The eigenvalues matrix of the Laplacian of the input data.
        """
        n_batches = (X.shape[0] // self.spectral_batch_size)
        Lambda = self._get_lambda_on_batch(self.spectral_batch_size)
        for i in range(1, n_batches):
            Lambda += self._get_lambda_on_batch()
        Lambda = torch.Tensor(Lambda) / n_batches
        return (Lambda + Lambda.T) / 2

    def _get_lambda_on_batch(self, batch_size=None) -> np.ndarray:
        """Get the eigenvalues of the Laplacian of a small batch of the input data.

        Returns
        -------
        np.ndarray
            The eigenvalues of the Laplacian of a small batch of the input data.
        """

        if batch_size is None:
            batch_size = self.spectral_batch_size
        batch_raw, batch_encoded = self._spectralnet.get_random_batch(batch_size)
        L_batch = self._get_laplacian_of_small_batch(batch_encoded)
        V_batch = self._predict(batch_raw)
        V_batch = V_batch / np.linalg.norm(V_batch, axis=0)
        Lambda = V_batch.T @ L_batch @ V_batch
        return Lambda
