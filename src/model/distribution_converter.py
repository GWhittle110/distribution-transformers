"""
Models for encoding arbitrary parametric distributions as GMMs
"""

import torch
from torch import nn
from typing import Optional, Union, Callable

from model.components import MixtureReshaper, MLP
from model.distribution_embeddings import DistributionEmbedding


class LatentDistributionConverter(nn.Module):

    def __init__(self, d_model: int, n_components: int):
        """
        Learnable mapping from latent representation of distribution to sequence of representations of GMM components.

        Args:
            d_model: Model size.
            n_components: Number of mixture components.

        """
        super().__init__()
        self.d_model = d_model
        self.n_components = n_components


class TransformerLatentDistributionConverter(LatentDistributionConverter):

    def __init__(self, d_model: int, n_components: int, n_head: int = 8, num_layers: int = 6,
                 dim_feedforward: Optional[int] = None, dropout: float = 0,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "gelu", **kwargs):
        """
        Learnable mapping from latent representation of distribution to sequence of representations of GMM components.

        Args:
            d_model: Model size.
            n_components: Number of mixture components.
            n_head: Number of attention heads per transformer layer.
            num_encoder_layers: Number of transformer encoder layers.
            dim_feedforward: Dimensionality of feedforward network model in each transformer layer.
                Defaults to 4 * d_model, as per the paper Attention is All You Need.
            dropout: Level of dropout.
                Defaults to 0.
            activation: Activation function of feedforward network model. "relu", "gelu" or a callable.
                Defaults to "gelu".

        """
        super().__init__(d_model, n_components)
        self.linear_projection = nn.Linear(d_model, n_components * d_model)
        encoder_layer = nn.TransformerEncoderLayer(nhead=n_head, d_model=d_model, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation=activation, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call.

        Args:
            x: Input latent representation, batch_shape X d_model.

        Returns:
            Output latent representation sequence, batch_shape X n_components X d_model.

        """
        x = self.linear_projection(x)
        x = x.reshape(x.shape[:-1] + (self.n_components, self.d_model))
        x = self.transformer_encoder(x)
        return x

    def init_weights(self):
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)


class MLPLatentDistributionConverter(LatentDistributionConverter):

    def __init__(self, d_model: int, n_components: int, hidden_layer_sizes: list[int]):
        """
        Learnable mapping from latent representation of distribution to sequence of representations of GMM components.

        Args:
            d_model: Model size.
            n_components: Number of mixture components.
            hidden_layer_sizes: Hidden layer sizes of MLPs

        """
        super().__init__(d_model, n_components)
        layer_sizes = [d_model] + hidden_layer_sizes + [d_model]
        self.mlp_list = nn.ModuleList([MLP(layer_sizes) for _ in range(n_components)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call.

        Args:
            x: Input latent representation, batch_shape X d_model.

        Returns:
            Output latent representation sequence, batch_shape X n_components X d_model.

        """
        x = torch.cat([mlp(x) for mlp in self.mlp_list], dim=-1)
        x = x.reshape(x.shape[:-1] + (self.n_components, self.d_model))
        return x


class ResidualLatentDistributionConverter(LatentDistributionConverter):

    def __init__(self, d_model: int, n_components: int, hidden_layer_sizes: list[int]):
        """
        Learnable mapping from latent representation of distribution to sequence of representations of GMM components.

        Args:
            d_model: Model size.
            n_components: Number of mixture components.
            hidden_layer_sizes: Hidden layer sizes of MLPs

        """
        super().__init__(d_model, n_components)
        layer_sizes = [d_model] + hidden_layer_sizes + [d_model]
        self.mlp_list = nn.ModuleList([MLP(layer_sizes) for _ in range(n_components)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call.

        Args:
            x: Input latent representation, batch_shape X d_model.

        Returns:
            Output latent representation sequence, batch_shape X n_components X d_model.

        """
        x = torch.cat([mlp(x)+x for mlp in self.mlp_list], dim=-1)
        x = x.reshape(x.shape[:-1] + (self.n_components, self.d_model))
        return x


class MLPTransformerLatentDistributionConverter(LatentDistributionConverter):

    def __init__(self, d_model: int, n_components: int, hidden_layer_sizes: list[int], n_head: int = 8,
                 num_layers: int = 6, dim_feedforward: Optional[int] = None, dropout: float = 0,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "gelu", **kwargs):
        """
        Learnable mapping from latent representation of distribution to sequence of representations of GMM components.

        Args:
            d_model: Model size.
            n_components: Number of mixture components.
            hidden_layer_sizes: Hidden layer sizes of MLPs
            n_head: Number of attention heads per transformer layer.
            num_encoder_layers: Number of transformer encoder layers.
            dim_feedforward: Dimensionality of feedforward network model in each transformer layer.
                Defaults to 4 * d_model, as per the paper Attention is All You Need.
            dropout: Level of dropout.
                Defaults to 0.
            activation: Activation function of feedforward network model. "relu", "gelu" or a callable.
                Defaults to "gelu".

        """
        super().__init__(d_model, n_components)
        layer_sizes = [d_model] + hidden_layer_sizes + [d_model]
        self.mlp_list = nn.ModuleList([MLP(layer_sizes) for _ in range(n_components)])
        encoder_layer = nn.TransformerEncoderLayer(nhead=n_head, d_model=d_model, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation=activation, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call.

        Args:
            x: Input latent representation, batch_shape X d_model.

        Returns:
            Output latent representation sequence, batch_shape X n_components X d_model.

        """
        x = torch.cat([mlp(x) for mlp in self.mlp_list], dim=-1)
        x = x.reshape(x.shape[:-1] + (self.n_components, self.d_model))
        x = self.transformer_encoder(x)
        return x

    def init_weights(self):
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)


class DistributionConverter(nn.Module):

    def __init__(self, distribution_embedding: DistributionEmbedding, latent_converter: LatentDistributionConverter,
                 component_embedding: DistributionEmbedding, mixture_reshaper: MixtureReshaper,
                 transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        """
        Convert from an arbitrary parametric distribution to a mixture model approximation via a latent space.

        Args:
            distribution_embedding: Embedding of parametric distribution.
            latent_converter: Converter model in latent space.
            component_embedding: Embedding of mixture components.
            mixture_reshaper: Reshaper from flat mixture representation to sequence representation.
            transform: Transform from sample space of parametric distribution to sample space of mixture model.
                Defaults to None.

        """
        super().__init__()
        self.distribution_embedding = distribution_embedding
        self.latent_converter = latent_converter
        self.component_embedding = component_embedding
        self.mixture_reshaper = mixture_reshaper
        self.transform = nn.Identity() if transform is None else transform

    def forward(self, phi_in: torch.Tensor) -> torch.Tensor:
        """
        Forward call.

        Args:
            phi_in: Parameters of parametric distribution.

        Returns:
            phi_out: Parameters of mixture model approximation.

        """
        phi_in_embedded = self.distribution_embedding(phi_in)
        phi_out_embedded = self.latent_converter(phi_in_embedded)
        phi_out_raw = self.component_embedding(phi_out_embedded, reverse=True)
        phi_out = self.mixture_reshaper(phi_out_raw, reverse=True)
        return phi_out
