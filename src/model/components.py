"""
Components for use in NN architectures
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Distribution
from torch.distributions.utils import vec_to_tril_matrix

from typing import Optional, Sequence, Union

from distributions.distributions import GaussianMixtureModel


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
        diag = torch.exp(diag)
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


class Logit(nn.Module):

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """
        Convert from weight space (0, 1) to logit space R.

        Args:
            x: Input tensor in (0, 1).

        Returns:
            Logit representation of x.

        """
        assert (x > 0).all() and (x < 1).all(), "Input must be in (0, 1)"
        return torch.log(x / (1 - x))


class MLP(nn.Sequential):

    def __init__(self, layer_sizes: Sequence[int], activation: Union[str, nn.Module] = "gelu"):
        """
        Simple MLP Block
        Args:
            layer_sizes: Sequence of layer sizes, including input and output sizes.
            activation: Activation function. Must be one of "relu", "gelu" or a callable.
                Defaults to "gelu"
        """
        super().__init__()
        match activation:
            case "relu":
                activation = nn.ReLU()
            case "gelu":
                activation = nn.GELU()
            case None:
                activation = nn.GELU()
            case _:
                raise ValueError('activation must be one of "relu", "gelu" or a nn.Module subclass')

        super().__init__(*sum([[activation, nn.Linear(size_in, size_out)]
                               for size_in, size_out in zip(layer_sizes[1:-1], layer_sizes[2:])],
                              start=[nn.Linear(layer_sizes[0], layer_sizes[1])]))


class MixtureReshaper(nn.Module):

    def __init__(self):
        """
        Convert mixture distributions between flat and sequence representations.
        """
        super().__init__()
        self.distribution: type[Distribution]

    def forward(self, phi: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            phi: Input parameters.
            reverse: Whether to run in reverse mode (sequence to flat).

        Returns:
            Reshaped parameters.

        """
        raise NotImplementedError

    def decode_sample(self, sample: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Decode tensor of sampled parameters to dictionary of tensors keyed by mixture parameter.

        Args:
            sample: Sampled tensor.

        Returns:
            Decoded sample.
        """
        raise NotImplementedError

    def encode_sample(self, decoded_sample: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode dictionary of sampled parameters to a singular tensor. Inverse operation of decode_sample.

        Args:
            decoded_sample:  Dictionary of decoded sample.

        Returns:
            Tensor encoding sample.
        """
        raise NotImplementedError


class GMMReshaper(MixtureReshaper):

    def __init__(self, n_components: int, state_size: int, logit_weights: bool = False):
        """
        Convert GMMs between flat and sequence representations.

        Args:
            n_components: Number of mixture components.
            state_size: Size of state in mixture model.
            logit_weights: Whether to move from weight space in flat representation to logit space in sequence
                representation.
                Defaults to False.

        """
        super().__init__()
        self.n_components = n_components
        self.state_size = state_size
        self.logit_weights = logit_weights
        self.distribution = GaussianMixtureModel

    def forward(self, phi: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            phi: Input parameters.
            reverse: Whether to run in reverse mode (sequence to flat).

        Returns:
            Reshaped parameters.

        """

        if not reverse:
            batch_shape = phi.shape[:-1]
            phi_w = phi[..., :self.n_components]
            if self.logit_weights:
                phi_w = torch.log(phi_w / (1 - phi_w))  # Move to logit space
            phi_w = phi_w.reshape(batch_shape + (self.n_components, 1))
            phi_mu = phi[..., self.n_components:self.n_components + self.state_size * self.n_components]
            phi_mu = phi_mu.reshape(batch_shape + (self.n_components, self.state_size))
            phi_scale = phi[..., -self.state_size ** 2 * self.n_components:]
            phi_scale = phi_scale.reshape(batch_shape + (self.n_components, self.state_size ** 2))
            phi = torch.cat([phi_w, phi_mu, phi_scale], dim=-1)

        else:
            batch_shape = phi.shape[:-2]
            phi_w = phi[..., :, 0].reshape(batch_shape + (self.n_components,))
            if self.logit_weights:
                phi_w = F.softmax(phi_w, dim=-1)
            phi_mu = phi[..., 1:1 + self.state_size].reshape(batch_shape + (self.n_components * self.state_size,))
            phi_scale = phi[..., -self.state_size ** 2:].reshape(batch_shape +
                                                                 (self.n_components * self.state_size ** 2,))
            phi = torch.cat([phi_w, phi_mu, phi_scale], dim=-1)
        return phi

    def decode_sample(self, sample: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Decode tensor of sampled parameters to dictionary of tensors keyed by GMM parameter.

        Args:
            sample: Sampled tensor.

        Returns:
            Decoded sample.
        """
        sample_shape = sample.shape[:-1]
        weights = sample[..., :self.n_components]
        loc = sample[..., self.n_components:self.n_components * (1 + self.state_size)].reshape(*sample_shape,
                                                                                               self.n_components,
                                                                                               self.state_size)
        scale = sample[..., -self.n_components * self.state_size ** 2:].reshape(*sample_shape, self.n_components,
                                                                                self.state_size, self.state_size)
        return {"weights": weights,
                "loc": loc,
                "covariance_matrix": scale}

    def encode_sample(self, decoded_sample: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode dictionary of sampled parameters to a singular tensor. Inverse operation of decode_sample.

        Args:
            decoded_sample:  Dictionary of decoded sample.

        Returns:
            Tensor encoding sample.
        """
        weights = decoded_sample["weights"]
        loc = decoded_sample["loc"]
        scale = decoded_sample["covariance_matrix"]
        return torch.cat([weights, loc.flatten(start_dim=-2), scale.flatten(start_dim=-3)], dim=-1)
