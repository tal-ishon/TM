import torch
import torch.nn as nn


class SpectralNetLoss(nn.Module):
    def __init__(self, laplacian_kind: str='rw'):
        super(SpectralNetLoss, self).__init__()
        self.laplacian_kind = laplacian_kind

    def forward(
        self, W: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        """
        This function computes the loss of the SpectralNet model.
        The loss is the rayleigh quotient of the Laplacian matrix obtained from W,
        and the orthonormalized output of the network.

        Args:
            W (torch.Tensor):               Affinity matrix
            Y (torch.Tensor):               Output of the network
            laplacian_kind (bool, optional): Specifies the kind of Laplacian matrix to use. Defaults to RW.

        Returns:
            torch.Tensor: The loss
        """
        m = Y.size(0)
        # if self.is_normalized:
        #     D = torch.sum(W, dim=1)
        #     Y = Y / torch.sqrt(D)[:, None]

        # Dy = torch.cdist(Y, Y)
        # loss = torch.sum(W * Dy.pow(2)) / (2 * m)

        D = torch.diag(torch.sum(W, dim=1)).to(W.device)
        if self.laplacian_kind == "unnormalized":
            L = D - W
        elif self.laplacian_kind == "rw":
            L = torch.eye(m).to(W.device) - torch.inverse(D) @ W
        else: # self.laplacian_kind == "symmetric"
            D_inv_sqrt = torch.sqrt(torch.inverse(D))
            L = torch.eye(m).to(W.device) - D_inv_sqrt @ W @ D_inv_sqrt

        loss = torch.trace(Y.T @ L @ Y) / m

        return loss
