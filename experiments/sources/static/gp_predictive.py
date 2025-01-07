"""
Experiment evaluating method on problem of finding posterior for GP hyperparameters
"""

from typing import Optional
import torch
from torch import Tensor
from torch.distributions import Normal, MultivariateNormal, constraints, Distribution, Uniform
from torch.distributions.utils import lazy_property
from torch.types import _size

from gpytorch import add_jitter
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean


from distributions.distributions import (InverseGammaMetaPrior, ObservationModel, CompleteDistribution,
                                         GaussianMixtureModel, MetaPrior)
from model.distribution_transformer import DistributionTransformer
from workflows.train import train
from workflows.test import test_conjugate_prior
from model.embeddings import ComponentEmbedding, GammaEmbedding, ObservationEmbedding

class MeanScaleMetaPrior(MetaPrior):
    def __init__(self, *args, **kwargs):
        super().__init__(prior=GaussianProcessPrior)

        self.metapriors = {
            "weights": torch.ones,
            "constant_mean": Uniform(-1, 1), 
            "output_scale": Uniform(0.5, 1.5),
        }

        self.metapriors_keylist = ["weights", "constant_mean", "output_scale"]
    
    def decode_sample(self, sample: Tensor) -> dict[str, Tensor]:
        return {k: sample[..., ix] for ix, k in enumerate(self.metapriors_keylist)}

    def encode_sample(self, decoded_sample: dict[str, Tensor]) -> Tensor:
        return torch.Tensor([decoded_sample[k] for k in self.metapriors_keylist])
    
    def sample(self, sample_shape):

        sampled_values = []

        for metaprior in self.metapriors.values():
            if isinstance(metaprior, Distribution):
                sampled_values.append(
                    metaprior.sample(sample_shape)
                )
            else:
                sampled_values.append(
                    metaprior(sample_shape)
                )
        

        return torch.stack(sampled_values, dim=-1).unsqueeze(-2)


class GaussianProcessPrior(Distribution):
    arg_constraints = {
            "constant_mean": constraints.real,
            "output_scale": constraints.positive,
            "weights": constraints.positive
        }

    def __init__(self, 
                 constant_mean: Tensor = torch.zeros(torch.Size()), 
                 output_scale: Tensor = torch.ones(torch.Size()),
                 weights: Tensor = torch.ones(torch.Size())
        
        ):
        super().__init__()
        self.dataset_size = torch.randint(5, 15, [1]).item()
        self.constant_mean = constant_mean
        self.output_scale = output_scale
        self.weights = weights

        assert self.output_scale.shape == self.constant_mean.shape
        self.hyperparams_batch_shape = self.output_scale.shape

        self.n_observations = self.dataset_size + 1

        self.x_distribution = Uniform(0, 5)
        self.kernel_lengthscale = 0.5


    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        kernel = ScaleKernel(RBFKernel(), batch_shape=self.hyperparams_batch_shape)
        kernel.base_kernel.lengthscale = self.kernel_lengthscale
        kernel.outputscale = self.output_scale

        mean = ConstantMean(batch_shape=self.hyperparams_batch_shape)
        mean.constant = self.constant_mean

        x = self.x_distribution.sample((self.n_observations,))
        y = MultivariateNormal(loc=mean(x),
                               covariance_matrix=add_jitter(kernel(x).to_dense())).sample()
        return torch.cat(
            [
                x.view(1, 1, 1, self.n_observations).expand(self.hyperparams_batch_shape + (self.n_observations,)).transpose(-1,-2), 
                y.transpose(-1,-2)
            ],
            dim=-1
        )


class GPPredictiveObservationModel(ObservationModel):
    def __init__(self, observation_type:str):
        super().__init__()

        assert observation_type in ["dataset", "query"]

        self.observation_type = observation_type
    
    def condition_(self, full_dataset):
        self.full_dataset = full_dataset
    
    def sample(self, sample_shape = ...):
        if self.observation_type == "dataset":
            return self.full_dataset[...,:-1]
        elif self.observation_type == "query":
            return self.full_dataset[...,-1, 0].unsqueeze(-1)


class CompleteDistributionGPPredictive(CompleteDistribution):
    def __init__(self, meta_prior, **observation_model):
        super().__init__(meta_prior, **observation_model)
    
    def sample(self,
               sample_shape: _size = torch.Size(),
               cache_prior: bool = False
               ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if not cache_prior or self.prior_sample is None:
            phi = self.meta_prior.sample(sample_shape)
            self.prior_sample = phi
        else:
            phi = self.prior_sample
        phi_decoded = self.meta_prior.decode_sample(phi)
        full_dataset = self.prior(**phi_decoded).sample()
        for observation_model in self.observation_model.values():
            observation_model.condition_(full_dataset)
        z = {key: observation_model.sample() for key, observation_model in self.observation_model.items()}

        unknown_y = full_dataset[...,-1,1].unsqueeze(-1)

        return phi, unknown_y, z

def run(n_components: int,
        state_size: int,
        meta_prior_kwargs: dict,
        component_embedding_kwargs: dict,
        observation_embedding_kwargs: dict[str, dict],
        transformer_kwargs: dict,
        training_kwargs: dict,
        testing_kwargs: dict,
        _run=None,
        *args, **kwargs):
    
    meta_prior = MeanScaleMetaPrior(**meta_prior_kwargs)

    observation_model = {observation_type: GPPredictiveObservationModel(observation_type=observation_type)
                         for observation_type in ["dataset", "query"]}

    # Complete distribution
    complete_distribution = CompleteDistributionGPPredictive(meta_prior, **observation_model)

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
