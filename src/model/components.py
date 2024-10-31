"""
Components for use in NN architectures
"""

import torch
from torch import nn
from torch.distributions.utils import vec_to_tril_matrix

from typing import Optional


class Cholesky(nn.Module):

    def __init__(self, state_size: int, jitter: Optional[float] = None):
        """
        Convert input to a lower-triangular (Cholesky) matrix, then recast to input dims.

        Args:
            state_size: Size of Cholesky matrix.
            jitter: Magnitude of jitter to add to diagonal of Cholesky matrix for conditioning.
                Defaults to 1e-6.

        """
        super().__init__()
        if jitter is None:
            jitter = 1e-6
        self.state_size = state_size
        self.jitter = jitter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Cholesky matrix recast to dimensionality of input tensor.

        """
        shape = x.shape
        if shape[-2:] == (self.state_size, self.state_size):
            diag = torch.diagonal(x, dim1=-2, dim2=-1)
            tril = torch.tril(x, -1)
        elif shape[-1] == self.state_size**2:
            x = x.reshape(shape[:-1] + (self.state_size, self.state_size))
            diag = torch.diagonal(x, dim1=-2, dim2=-1)
            tril = torch.tril(x, -1)
        elif shape[-1] == self.state_size * (self.state_size + 1) // 2:
            x = vec_to_tril_matrix(x)
            diag = torch.diagonal(x, dim1=-2, dim2=-1)
            tril = torch.tril(x, -1)
        else:
            raise ValueError("Cannot cast input as lower triangular matrix")
        diag = torch.abs(diag)
        diag = diag + self.jitter
        tril_complete = tril + torch.diag_embed(diag, dim1=-2, dim2=-1)
        return tril_complete.reshape(shape[:-1] + (-1,))


class PositiveDefinite(Cholesky):

    def __init__(self, state_size: int, jitter: Optional[float] = None):
        """
        Convert input to a positive definite matrix, then recast to input dims

        Args:
            state_size: Size of square matrix.
            jitter: Magnitude of jitter to add to diagonal of Cholesky decomposition of matrix. Note that this
                contributes to the square root of the determinant of the resulting positive definite matrix.
                Defaults to 1e-3.
        """
        jitter = 1e-3 if jitter is None else jitter
        super().__init__(state_size, jitter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor

        Returns:
            Positive definite matrix recast to dimensionality of input tensor

        """
        shape = x.shape
        chol = super().forward(x)
        if shape[-2:] == (self.state_size, self.state_size):
            chol_mat = chol
        else:
            chol_mat = chol.reshape(shape[:-1] + (self.state_size, self.state_size))
        mat = torch.einsum("...ij, ...kj -> ...ik", chol_mat, chol_mat)
        return mat.reshape(chol.shape)
