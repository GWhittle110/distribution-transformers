"""
Transformer model architectures for each prior/posterior distribution parametrisation
"""

import torch
from torch import nn
import torch.nn.functional as F

from typing import Union, Callable

from model.components import Cholesky, PositiveDefinite, Logit
from model.distribution_embeddings import GaussianEmbedding


class TransformerModel(nn.Module):
    """
    Abstract base class of transformer models
    """

    def forward(self, phi: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transformer model.
        Args:
            phi: Batched prior parameter Tensor of form [weights | locs | scales]
            z: Batched observation Tensor

        Returns:
            Posterior parameter Tensor.

        """


class GMMTransformerModel(TransformerModel):
    def __init__(self, n_components: int, state_size: int, n_observations: int, d_model: int, n_head: int,
                 scale_parametrisation: str = None, num_transformer_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.gelu, **kwargs):
        """
        Transformer model for dealing with Gaussian mixture model priors and posteriors. Generates tokens from
        concatenated Tensor of prior parameters and observations. Deprecated.

        Args:
            n_components: Number of components in Gaussian mixture model.
            state_size: Size of the Gaussian mixture model's state.
            n_observations: Number/size of observations.
            d_model: Number of features in input to transformer layer.
            n_head: Number of self-attention heads in transformer.
            scale_parametrisation: Parametrisation of scale for Gaussian mixture model.
                Defaults to covariance_matrix.
            num_transformer_layers: Number of transformer layers in model.
                Defaults to 6.
            dim_feedforward: Dimensionality of feedforward network model in each transformer layer.
                Defaults to 2048.
            dropout: Level of dropout.
                Defaults to 0.1.
            activation: Activation function of feedforward network model. "relu", "gelu" or a callable.
                Defaults to "gelu".
            **kwargs: Additional keyword arguments for the transformer encoder layer.

        """
        super().__init__()
        self.n_components = n_components
        self.state_size = state_size
        self.n_parameters = n_components * (1 + state_size + state_size ** 2)
        self.n_observations = n_observations
        self.scale_parametrisation = "covariance_matrix" if scale_parametrisation is None else scale_parametrisation
        self.n_in = self.n_parameters + n_observations
        self.n_out = self.n_parameters
        self.d_model = d_model
        self.nhead = n_head
        self.feedforward_in = nn.Linear(self.n_in, d_model * n_head)
        transformer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=dim_feedforward, dropout=dropout,
                                                 activation=activation, batch_first=True, **kwargs)
        self.transformer_encoder = nn.TransformerEncoder(transformer, num_transformer_layers)
        self.feedforward_out = nn.Linear(self.d_model, self.n_out)
        match self.scale_parametrisation:
            case "covariance_matrix":
                self.scale_transform = PositiveDefinite(state_size)
            case "precision_matrix":
                self.scale_transform = PositiveDefinite(state_size)
            case "scale_tril":
                self.scale_transform = Cholesky(state_size)
            case _:
                raise AssertionError('scale_parametrisation must be one of "covariance_matrix", "precision_matrix" or '
                                     '"scale_tril"')
        self.init_weights()

    def forward(self, phi: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch_shape = phi.shape[:-1]
        x = torch.cat([phi, z], dim=-1)
        x = self.feedforward_in(x)
        x = x.reshape(batch_shape + (self.nhead, self.d_model))
        x = self.transformer_encoder(x)
        x = x[..., -1, :]
        x = self.feedforward_out(x)
        phi_w = F.softmax(x[..., :self.n_components], dim=-1)
        phi_mu = x[..., self.n_components:self.n_components + self.state_size * self.n_components]
        raw_scale = x[..., self.n_components + self.state_size * self.n_components:]
        scale_shape = raw_scale.shape
        phi_scale = self.scale_transform(raw_scale.reshape(raw_scale.shape[:-1] + (self.n_components,
                                                                                   self.state_size, self.state_size))
                                         ).reshape(scale_shape)
        phi = torch.cat([phi_w, phi_mu, phi_scale], dim=-1)
        return phi

    def init_weights(self):
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)


class GMMConditionalTransformerModel(TransformerModel):
    def __init__(self, n_components: int, state_size: int, n_observations: int, d_model: int, n_head: int,
                 scale_parametrisation: str = None, num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = None, dropout: float = 0.,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.gelu, **kwargs):
        """
        Conditional Transformer model for dealing with Gaussian mixture model priors and posteriors.
        Treats the Gaussian mixture model as a permutation-invariant sequence of (weight, Gaussian density) pairs,
        and conditions the output sequence on the encoded and self-attended sequence of observations using a transformer
        decoder. Uses prior component encodings that respect the information geometry of the component densities.

        Args:
            n_components: Number of components in Gaussian mixture model.
            state_size: Size of the Gaussian mixture model's state.
            n_observations: Number/size of observations.
            d_model: Number of features in input to transformer layer.
            n_head: Number of self-attention heads in transformer.
            scale_parametrisation: Parametrisation of scale for Gaussian mixture model.
                Defaults to covariance_matrix.
            num_encoder_layers: Number of transformer encoder layers in model.
                Defaults to 6.
            num_decoder_layers: Number of transformer decoder layers in model.
                Defaults to 6.
            dim_feedforward: Dimensionality of feedforward network model in each transformer layer.
                Defaults to 4 * d_model, as per the paper Attention is All You Need.
            dropout: Level of dropout.
                Defaults to 0.
            activation: Activation function of feedforward network model. "relu", "gelu" or a callable.
                Defaults to "gelu".
            **kwargs: Additional keyword arguments for the transformer encoder and decoder layers.
        """
        super().__init__()
        self.n_components = n_components
        self.state_size = state_size
        self.parameters_per_component = 1 + state_size + state_size ** 2
        self.n_observations = n_observations
        self.scale_parametrisation = "covariance_matrix" if scale_parametrisation is None else scale_parametrisation
        self.d_model = d_model
        self.nhead = n_head
        self.weight_transform = Logit()
        dim_feedforward = 4 * d_model if dim_feedforward is None else dim_feedforward

        self.gaussian_embedding = GaussianEmbedding(state_size=state_size,
                                                    d_model=d_model,
                                                    hidden_layer_sizes=[d_model // 2])
        self.observation_embedding = nn.Sequential(nn.Linear(self.n_observations, d_model // 2),
                                                   nn.GELU(),
                                                   nn.Linear(d_model // 2, d_model),
                                                   nn.GELU(),
                                                   nn.Linear(d_model, d_model))

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=dim_feedforward,
                                                               dropout=dropout, activation=activation,
                                                               batch_first=True, **kwargs)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_encoder_layers)

        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model, n_head, dim_feedforward=dim_feedforward,
                                                               dropout=dropout, activation=activation,
                                                               batch_first=True, **kwargs)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_decoder_layers)

        self.init_weights()

    def forward(self, phi_out: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch_shape = phi_out.shape[:-1]
        phi_in_w = self.weight_transform(phi_out[..., :self.n_components])    # Move to logit space
        phi_in_w = phi_in_w.reshape(batch_shape + (self.n_components, 1))
        phi_in_mu = phi_out[..., self.n_components:self.n_components + self.state_size * self.n_components]
        phi_in_mu = phi_in_mu.reshape(batch_shape + (self.n_components, self.state_size))
        phi_in_scale = phi_out[..., -self.state_size ** 2 * self.n_components:]
        phi_in_scale = phi_in_scale.reshape(batch_shape + (self.n_components, self.state_size ** 2))
        phi_in = torch.cat([phi_in_w, phi_in_mu, phi_in_scale], dim=-1)
        phi_in_embedded = self.gaussian_embedding(phi_in)
        z_embedded = self.observation_embedding(z).unsqueeze(-2)
        encoder_output = self.transformer_encoder(z_embedded)
        decoder_output = self.transformer_decoder(phi_in_embedded, encoder_output, tgt_is_causal=False)
        phi_out_raw = self.gaussian_embedding(decoder_output, reverse=True)
        phi_out_w_raw = phi_out_raw[..., :, 0].reshape(batch_shape + (self.n_components,))
        phi_out_w = F.softmax(phi_out_w_raw, dim=-1)
        phi_out_mu = phi_out_raw[..., 1:1 + self.state_size].reshape(batch_shape +
                                                                     (self.n_components * self.state_size,))
        phi_out_scale = phi_out_raw[..., -self.state_size ** 2:].reshape(batch_shape +
                                                                         (self.n_components * self.state_size ** 2,))
        phi_out = torch.cat([phi_out_w, phi_out_mu, phi_out_scale], dim=-1)
        return phi_out

    def init_weights(self):
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)
        for layer in self.transformer_decoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)
