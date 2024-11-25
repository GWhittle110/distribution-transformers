"""
Information geometry-respecting learnable embeddings of mixture model component distributions
"""

import torch
from torch import nn
from torch.distributions.utils import vec_to_tril_matrix

from typing import Union, Sequence


class DistributionEmbedding(nn.Module):
    """
    Abstract base class for all distribution embeddings.
    """

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        """
        Forward call of module.

        Args:
            x: Flattened tensor of component weight (as logit) and parameters | embedded component.
            reverse: Whether to run in forward (parameters to embedding) or reverse (embedding to parameters) mode.
                Note that the parametrisation of weights
                Defaults to False.

        Returns:
            Embedded component | flattened tensor of component weight (as logit) and parameters.

        """
        raise NotImplementedError


class GaussianEmbedding(DistributionEmbedding):

    def __init__(self, state_size: int, d_model: int, hidden_layer_sizes: Sequence[int] = None,
                 activation: Union[str, nn.Module] = None,
                 jitter: float = None, scale_parametrisation: str = None):
        """
        Learnable embedding of Gaussian components, that is to say a mapping from the manifold [0, 1] X SG to the
        model embedding space R**d_model. Takes in Gaussian components of the form [w | loc | scale.flatten()].

        Args:
            state_size: Dimensionality of component variable.
            d_model: Dimensionality of model embedding space.
            hidden_layer_sizes: Sequence of hidden layer sizes, if used.
                Defaults to None.
            activation: Activation function between hidden layers. "relu", "gelu" or a callable.
                Defaults to "gelu".
            jitter: Small value to add to diagonal of scale matrix for conditioning.
                Defaults to 1e-6
            scale_parametrisation: Parametrisation of Gaussian scale parameter.
                Must be one of "covariance_matrix", "precision_matrix" of "scale_tril".
                Defaults to "covariance_matrix"

        """
        super().__init__()
        self.state_size = state_size
        self.d_model = d_model
        self.jitter = 1e-6 if jitter is None else jitter
        assert scale_parametrisation in {"covariance_matrix" "precision_matrix", "scale_tril", None}, \
            'Scale parametrisation must be one of "covariance_matrix", "precision_matrix" of "scale_tril"'
        self.scale_parametrisation = "covariance_matrix" if scale_parametrisation is None else scale_parametrisation
        self.n_in = 1 + 2 * state_size + state_size * (state_size - 1) // 2

        match activation:
            case "relu":
                activation = nn.ReLU()
            case "gelu":
                activation = nn.GELU()
            case None:
                activation = nn.GELU()
            case _:
                raise ValueError('activation must be one of "relu", "gelu" or a nn.Module subclass')

        if hidden_layer_sizes is None:
            self.model_forward = nn.Linear(self.n_in, d_model)
            self.model_reverse = nn.Linear(d_model, self.n_in)
        else:
            self.model_forward = nn.Sequential(*sum([[nn.Linear(size_in, size_out), activation]
                                               for size_in, size_out in zip(hidden_layer_sizes[:-1],
                                                                            hidden_layer_sizes[1:])],
                                               start=[nn.Linear(self.n_in, hidden_layer_sizes[0]), activation]),
                                               nn.Linear(hidden_layer_sizes[-1], d_model))
            self.model_reverse = nn.Sequential(*sum([[nn.Linear(size_in, size_out), activation]
                                               for size_in, size_out in zip(hidden_layer_sizes[:0:-1],
                                                                            hidden_layer_sizes[-2::-1])],
                                               start=[nn.Linear(d_model, hidden_layer_sizes[-1])]),
                                               nn.Linear(hidden_layer_sizes[0], self.n_in))

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        """
        Forward call of module.

        Args:
            x: Flattened tensor of component weight (as logit) and parameters | embedded component.
            reverse: Whether to run in forward (parameters to embedding) or reverse (embedding to parameters) mode.
                Note that the parametrisation of weights
                Defaults to False.

        Returns:
            Embedded component | flattened tensor of component weight (as logit) and parameters.

        """
        batch_shape = x.shape[:-1]
        n_scale = self.state_size ** 2
        n_scale_flat = self.state_size + self.state_size * (self.state_size - 1) // 2
        if reverse:
            # Reverse mode
            x = self.model_reverse(x)
            scale_flat = x[..., -n_scale_flat:]
            diag = scale_flat[..., :self.state_size].exp() + self.jitter
            scale = vec_to_tril_matrix(scale_flat[..., self.state_size:], -1) + torch.diag_embed(diag)
            if self.scale_parametrisation != "scale_tril":
                scale = torch.einsum("...ij,...kj->...ik", scale, scale)
            return torch.cat([x[..., :-n_scale_flat], scale.reshape(batch_shape + (n_scale,))], dim=-1)

        else:
            # Forward mode
            scale = x[..., -n_scale:].reshape(batch_shape + (self.state_size, self.state_size))
            if self.scale_parametrisation != "scale_tril":
                scale = torch.linalg.cholesky(scale)
            diag = torch.diagonal(scale, dim1=-2, dim2=-1).log()
            scale_flat = torch.cat([diag,
                                    scale[..., *torch.tril_indices(self.state_size, self.state_size, -1).tolist()]],
                                   dim=-1)
            x = torch.cat([x[..., :-n_scale], scale_flat], dim=-1)
            x = self.model_forward(x)
            return x


class GammaEmbedding(DistributionEmbedding):

    def __init__(self, d_model: int, hidden_layer_sizes: Sequence[int] = None,
                 activation: Union[str, nn.Module] = None, no_weight: bool = True):
        """
        Learnable embedding of Gamma/Inverse Gamma distribution, that is to say a mapping from the manifold [0, 1], SGa
        to the model embedding space R**d_model. Takes in Gamma components of the form [concentration, rate].

        Args:
            d_model: Dimensionality of model embedding space.
            hidden_layer_sizes: Sequence of hidden layer sizes, if used.
                Defaults to None.
            activation: Activation function between hidden layers. "relu", "gelu" or a callable.
                Defaults to "gelu".
            no_weight: Whether to not include weight parameter.
                Defaults to True.

        """
        super().__init__()
        self.d_model = d_model
        self.n_in = 3 - no_weight
        self.no_weight = no_weight

        match activation:
            case "relu":
                activation = nn.ReLU()
            case "gelu":
                activation = nn.GELU()
            case None:
                activation = nn.GELU()
            case _:
                raise ValueError('activation must be one of "relu", "gelu" or a nn.Module subclass')

        if hidden_layer_sizes is None:
            self.model_forward = nn.Linear(self.n_in, d_model)
            self.model_reverse = nn.Linear(d_model, self.n_in)
        else:
            self.model_forward = nn.Sequential(*sum([[nn.Linear(size_in, size_out), activation]
                                               for size_in, size_out in zip(hidden_layer_sizes[:-1],
                                                                            hidden_layer_sizes[1:])],
                                               start=[nn.Linear(self.n_in, hidden_layer_sizes[0]), activation]),
                                               nn.Linear(hidden_layer_sizes[-1], d_model))
            self.model_reverse = nn.Sequential(*sum([[nn.Linear(size_in, size_out), activation]
                                               for size_in, size_out in zip(hidden_layer_sizes[:0:-1],
                                                                            hidden_layer_sizes[-2::-1])],
                                               start=[nn.Linear(d_model, hidden_layer_sizes[-1])]),
                                               nn.Linear(hidden_layer_sizes[0], self.n_in))

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        """
        Forward call of module.

        Args:
            x: Flattened tensor of component weight (as logit) and parameters | embedded component.
            reverse: Whether to run in forward (parameters to embedding) or reverse (embedding to parameters) mode.
                Note that the parametrisation of weights
                Defaults to False.

        Returns:
            Embedded component | flattened tensor of component weight (as logit) and parameters.

        """
        if reverse:
            # Reverse mode
            return self.model_reverse(x).exp()

        else:
            # Forward mode
            return self.model_forward(x.log())
