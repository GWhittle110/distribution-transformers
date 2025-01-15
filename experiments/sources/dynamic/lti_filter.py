"""
Experiment testing method on time series filtering
"""

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal, Exponential, Normal, Uniform, Categorical, constraints
from torch.distributions.utils import lazy_property

from typing import Optional

from torch.types import _size

from distributions.distributions import (GaussianMixtureModelConjugateMetaPrior,
                                         CompleteDistribution, ObservationModel)
from model.embeddings import ComponentEmbedding, ObservationEmbedding
from model.distribution_transformer import DistributionTransformer
from workflows.train import train
from workflows.test import test_lti_filter
from dynamic.motion_models import LTIMotionModel


class RangefinderObservationModel(ObservationModel):
    arg_constraints = {"scale": constraints.positive,
                       "rate": constraints.positive,
                       "max_range": constraints.positive,
                       "weights": constraints.simplex}

    def __init__(self, scale: Tensor,
                 rate: Tensor,
                 max_range: Tensor,
                 weights: Tensor):
        """
        Radar / Sonar rangefinder model. Comprised of mixture of Gaussian centered at true observation with standard
        deviation proportional to range + 1, uniform noise, modelling sensor failure, exponential noise representing
        unexpected interruptions and a maximum range term.

        Args:
            scale: Standard deviation of Gaussian component / range.
            rate: Decay constant of exponential component.
            max_range: Maximum sensor range.
            weights: Weights between Gaussian, exponential and uniform components.
        """
        super().__init__()
        self.scale = scale
        self.rate = rate
        self.max_range = max_range
        self.weights = weights

        self.mixture_distribution = Categorical(weights)
        self.exponential_distribution = Exponential(rate)
        self.uniform_distribution = Uniform(torch.tensor(0.), max_range)

        self.normal_distribution: Optional[Normal] = None

    def condition_(self, x: Tensor):
        self.device = x.device
        self.to_device()
        range = torch.sqrt(x[..., 0] ** 2 + x[..., 2] ** 2)
        self.normal_distribution = Normal(range, self.scale * (range + 1))

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        expanded_sample_shape = sample_shape + self.normal_distribution.batch_shape
        mix_sample = self.mixture_distribution.sample(expanded_sample_shape)
        normal_sample = self.normal_distribution.sample(sample_shape)
        exponential_sample = self.exponential_distribution.sample(expanded_sample_shape)
        uniform_sample = self.uniform_distribution.sample(expanded_sample_shape)
        samples = torch.stack([normal_sample, exponential_sample, uniform_sample], -1)

        mix_shape = mix_sample.shape
        mix_sample_r = mix_sample.unsqueeze(-1)
        mix_sample_r = mix_sample_r.repeat(torch.Size([1] * (len(mix_shape) + 1)))

        samples = samples.gather(-1, mix_sample_r)
        samples = torch.maximum(samples, torch.tensor(0.))
        samples = torch.minimum(samples, self.max_range)
        return samples

    def cdf(self, value: Tensor) -> Tensor:
        mix_prob = self.mixture_distribution.probs
        cdf_normal = self.normal_distribution.cdf(value)
        cdf_exponential = self.exponential_distribution.cdf(value)
        cdf_uniform = value / self.max_range
        cdf_stack = torch.stack([cdf_normal, cdf_exponential, cdf_uniform], dim=-1)
        cdf = torch.sum(cdf_stack * mix_prob, dim=-1)
        cdf[value >= self.max_range] = 1.
        return cdf

    def log_prob(self, value: Tensor) -> Tensor:
        log_mix_prob = torch.log_softmax(
            self.mixture_distribution.logits, dim=-1
        )
        log_normal_prob = self.normal_distribution.log_prob(value)
        log_exponential_prob = self.exponential_distribution.log_prob(value)
        log_uniform_prob = torch.log((value <= self.max_range) / self.max_range)
        log_probs = torch.stack([log_normal_prob, log_uniform_prob, log_exponential_prob], dim=-1)
        return torch.logsumexp(log_probs + log_mix_prob, dim=-1)

    @property
    def mean(self) -> Tensor:
        probs = self.mixture_distribution.probs
        mean_normal = self.normal_distribution.mean
        mean_exponential = self.exponential_distribution.mean.broadcast_to(mean_normal.shape)
        mean_uniform = self.uniform_distribution.mean.broadcast_to(mean_normal.shape)
        mean_stack = torch.stack([mean_normal, mean_exponential, mean_uniform], dim=-1)
        return torch.sum(mean_stack * probs, dim=-1)

    @property
    def variance(self) -> Tensor:
        probs = self.mixture_distribution.probs
        mean_normal = self.normal_distribution.mean
        mean_exponential = self.exponential_distribution.mean.broadcast_to(mean_normal.shape)
        mean_uniform = self.uniform_distribution.mean.broadcast_to(mean_normal.shape)
        mean_stack = torch.stack([mean_normal, mean_exponential, mean_uniform], dim=-1)
        var_cond_mean = torch.sum(probs * (mean_stack -
                                           self.mean.broadcast_to(3, *mean_normal.shape).swapdims(0, -1)) ** 2,
                                  dim=-1)

        mean_cond_var_normal = self.normal_distribution.variance.broadcast_to(mean_normal.shape)
        mean_cond_var_exponential = self.exponential_distribution.variance.broadcast_to(mean_normal.shape)
        mean_cond_var_uniform = self.uniform_distribution.variance.broadcast_to(mean_normal.shape)
        mean_cond_var_stack = torch.stack([mean_cond_var_normal, mean_cond_var_exponential,
                                           mean_cond_var_uniform], dim=-1)
        mean_cond_var = torch.sum(probs * mean_cond_var_stack, dim=-1)

        return mean_cond_var + var_cond_mean

    def to_device(self):
        device = self.device
        self.scale = self.scale.to(device)
        self.rate = self.rate.to(device)
        self.max_range = self.max_range.to(device)
        self.weights = self.weights.to(device)

        self.mixture_distribution = Categorical(self.weights)
        self.exponential_distribution = Exponential(self.rate)
        self.uniform_distribution = Uniform(torch.tensor(0., device=device), self.max_range)

    @lazy_property
    def scale(self):
        return self.scale

    @lazy_property
    def rate(self):
        return self.rate

    @lazy_property
    def max_range(self):
        return self.max_range

    @lazy_property
    def weights(self):
        return self.weights


class GaussianAngleObservationModel(ObservationModel):
    arg_constraints = {"scale": constraints.positive}

    def __init__(self, scale: Tensor):
        """
        Noisy angle sensor.

        Args:
            scale: Noise standard deviation.
        """
        super().__init__()
        self.scale = scale

    def condition_(self, x: Tensor):
        self.device = x.device
        self.scale = self.scale.to(x.device)
        angle = torch.atan2(x[..., 0], x[..., 2])
        self.distribution = Normal(angle, self.scale)

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        sample = self.distribution.sample(sample_shape)
        return ((sample + torch.pi) % (2 * torch.pi) - torch.pi).unsqueeze(-1)

    @lazy_property
    def scale(self):
        return self.scale


def run(n_components: int,
        state_size: int,
        meta_prior_kwargs: dict,
        observation_model_kwargs: dict[str, dict],
        component_embedding_kwargs: dict,
        observation_embedding_kwargs: dict[str, dict],
        transformer_kwargs: dict,
        motion_model_kwargs: dict,
        training_kwargs: dict,
        testing_kwargs: dict,
        _run=None,
        *args, **kwargs):
    """
    Run an experiment comparing distribution transformers to the closed form posterior of a GMM prior under linear
    Gaussian observations.

    Args:
        n_components: Number of GMM components.
        state_size: Dimensionality of GMM.
        meta_prior_kwargs: Dictionary of parameters for the meta prior.
        observation_model_kwargs: Dictionary of dictionaries of kwargs for observation models.
        component_embedding_kwargs: Dictionary of component embedding parameters.
        observation_embedding_kwargs: Dictionary of dictionaries of observation embedding parameters.
        transformer_kwargs: Dictionary of parameters for the transformer model.
        motion_model_kwargs: Dictionary of kwargs for the motion model.
        training_kwargs: Dictionary of parameters for the training routine.
        testing_kwargs: Dictionary of parameters for the testing routine.
        _run: Sacred run object.

    Returns:

    """

    # Meta-prior
    meta_prior = GaussianMixtureModelConjugateMetaPrior(state_size=state_size, n_components=n_components,
                                                        **meta_prior_kwargs)

    # Observation model
    observation_model = {
        "obs_1": RangefinderObservationModel(torch.tensor(observation_model_kwargs["obs_1"]["scale"]),
                                             torch.tensor(observation_model_kwargs["obs_1"]["rate"]),
                                             torch.tensor(observation_model_kwargs["obs_1"]["max_range"]),
                                             torch.tensor(observation_model_kwargs["obs_1"]["weights"])),
        "obs_2": GaussianAngleObservationModel(torch.tensor(observation_model_kwargs["obs_2"]["scale"]))
    }

    # Complete distribution
    complete_distribution = CompleteDistribution(meta_prior, **observation_model)

    # Distribution transformer
    d_model = transformer_kwargs["d_model"]
    component_embedding = ComponentEmbedding(state_size=state_size, d_model=d_model, **component_embedding_kwargs)
    observation_embedding = {key: ObservationEmbedding(d_model=d_model, observation_size=1, **kwargs)
                             for key, kwargs in observation_embedding_kwargs.items()}
    model = DistributionTransformer(component_embedding=component_embedding,
                                    transformer_kwargs=transformer_kwargs,
                                    n_components=n_components,
                                    prior_embedding=None,
                                    sample_space_transform=None,
                                    **observation_embedding)

    model, last_epoch_metrics = train(model, complete_distribution, _run=_run, **training_kwargs)

    motion_model = LTIMotionModel(torch.tensor(motion_model_kwargs["state_transition_matrix"]),
                                  torch.tensor(motion_model_kwargs["process_noise_scale_cholesky"]),
                                  MultivariateNormal(
                                      torch.tensor(motion_model_kwargs["x0_loc"]),
                                      torch.tensor(motion_model_kwargs["x0_covariance_matrix"])
                                  ))

    test_lti_filter(model, motion_model, observation_model, _run=_run, **testing_kwargs)
