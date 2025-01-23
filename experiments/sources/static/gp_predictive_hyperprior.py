"""
Experiment evaluating method on problem of finding posterior for GP hyperparameters
"""

from functools import partial
from random import sample
from typing import Union, Sequence, Callable, Optional
import gpytorch
from sympy import hyper
import torch
from torch import Tensor
from torch import nn
from torch.distributions import Normal, MultivariateNormal, constraints, Distribution, Uniform, Normal, Independent
from torch.distributions.utils import lazy_property
from torch.types import _size

from gpytorch import add_jitter
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean


from distributions.distributions import (InverseGammaMetaPrior, ObservationModel, CompleteDistribution,
                                         GaussianMixtureModel, MetaPrior)
from model.embeddings import DistributionEmbedding
from model.distribution_transformer import DistributionTransformer
from distributions.utils import gmm_bounds_func
from workflows.train import train
from workflows.test import test_gp
from model.embeddings import ComponentEmbedding, GammaEmbedding, ObservationEmbedding

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean_module, covar_module):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class GaussianEmbedding(DistributionEmbedding):

    def __init__(self,
                 d_model: int,
                 n_components: int,
                 embedding_hidden_layer_sizes: Optional[Sequence[int]] = None,
                 embedding_activation: Union[str, nn.Module] = nn.GELU,
                 conversion_hidden_layer_sizes: Optional[Sequence[int]] = None,
                 conversion_activation: Union[str, nn.Module] = nn.GELU):
        """
        Learnable embedding from Gamma/Inverse Gamma distributions to GMM representation in model latent space.
        Note that this embedding is not invertible.

        Args:
            d_model: Dimensionality of model latent space.
            n_components: Number of GMM components.
            embedding_hidden_layer_sizes: Sequence of hidden layer sizes in MLP embedding from transformed parameter
                space to model latent space, if used.
                Defaults to None
            embedding_activation: Activation function between hidden layers in MLP embedding from transformed parameter
                space to model latent space. "relu", "gelu" or a nn.Module subclass.
                Defaults to GELU.
            conversion_hidden_layer_sizes: Sequence of hidden layer sizes in MLP conversion to latent GMM
                representation, if used.
                Defaults to None.
            conversion_activation: Activation function between hidden layers in MLP conversion to latent GMM
                representation. "relu", "gelu" or a nn.Module subclass.
                Defaults to GELU.

        """
        super().__init__(n_params=2,
                         n_components=n_components,
                         d_model=d_model,
                         transform=lambda x: torch.stack([x[...,0], torch.log(x[...,1])], dim=-1),
                         embedding_hidden_layer_sizes=embedding_hidden_layer_sizes,
                         embedding_activation=embedding_activation,
                         conversion_hidden_layer_sizes=conversion_hidden_layer_sizes,
                         conversion_activation=conversion_activation)

class HyperpriorEmbedding(DistributionEmbedding):

    def __init__(self,
                 d_model: int,
                 n_components: int,
                 state_size: int,
                 embedding_hidden_layer_sizes: Optional[Sequence[int]] = None,
                 embedding_activation: Union[str, nn.Module] = nn.GELU,
                 conversion_hidden_layer_sizes: Optional[Sequence[int]] = None,
                 conversion_activation: Union[str, nn.Module] = nn.GELU,
                 **component_embedding_kwargs):
        """
        Learnable embedding from Gamma/Inverse Gamma distributions to GMM representation in model latent space.
        Note that this embedding is not invertible.

        Args:
            d_model: Dimensionality of model latent space.
            n_components: Number of GMM components.
            embedding_hidden_layer_sizes: Sequence of hidden layer sizes in MLP embedding from transformed parameter
                space to model latent space, if used.
                Defaults to None
            embedding_activation: Activation function between hidden layers in MLP embedding from transformed parameter
                space to model latent space. "relu", "gelu" or a nn.Module subclass.
                Defaults to GELU.
            conversion_hidden_layer_sizes: Sequence of hidden layer sizes in MLP conversion to latent GMM
                representation, if used.
                Defaults to None.
            conversion_activation: Activation function between hidden layers in MLP conversion to latent GMM
                representation. "relu", "gelu" or a nn.Module subclass.
                Defaults to GELU.

        """
        super().__init__(n_params=5, d_model=d_model, n_components=n_components)
        self.y_prior_embedding = GaussianEmbedding(
            d_model=d_model // 2, 
            n_components=n_components,
            embedding_hidden_layer_sizes=embedding_hidden_layer_sizes,
            embedding_activation=embedding_activation,
            conversion_hidden_layer_sizes=conversion_hidden_layer_sizes,
            conversion_activation=conversion_activation
        )
        self.lengthscale_prior_embedding = GammaEmbedding(
            d_model=d_model - d_model // 2, 
            n_components=n_components,
            embedding_hidden_layer_sizes=embedding_hidden_layer_sizes,
            embedding_activation=embedding_activation,
            conversion_hidden_layer_sizes=conversion_hidden_layer_sizes,
            conversion_activation=conversion_activation
        )

    def embed(self, x: Tensor) -> Tensor:
        """
        Embed input tensor into model latent space.

        Args:
            x: Input parameter tensor.

        Returns:
            Embedded GMM representation of input parameter tensor.

        """

        y_prior = x[...,:2] # batch_size x 2 (loc, scale)
        lengthscale_prior = x[...,2:] # batch_size x 2 (concentration, rate)
        
        y_emb = self.y_prior_embedding.embed(y_prior) # batch_size n_components x d_model // 2
        lengthscale_emb = self.lengthscale_prior_embedding.embed(lengthscale_prior) # batch_size x n_components x (d_model - d_model // 2)

        return torch.cat([y_emb, lengthscale_emb], dim=-1) # batch_size x (n_components + 1) x d_model

    def de_embed(self, x: Tensor) -> Tensor:
        """
        De_embedding not supported by DistributionEmbedding.

        """
        breakpoint()
        raise NotImplementedError



class MeanScaleMetaPrior(MetaPrior):
    def __init__(self, *args, **kwargs):
        '''
        Class for sampling the mean and scale paramaters for a GP prior
        '''
        super().__init__(prior=GaussianProcessPrior)

        self.metapriors_keylist = ["loc", "covariance_matrix", "lengthscale_prior_concentration", "lengthscale_prior_rate"]
    
        self.metapriors = {
            "loc": Uniform(kwargs.get("constant_mean_low"), kwargs.get("constant_mean_high")), 
            "covariance_matrix": Uniform(kwargs.get("output_scale_low"), kwargs.get("output_scale_high")),
            "lengthscale_prior_concentration": Uniform(kwargs.get("lengthscale_prior_concentration_low"), kwargs.get("lengthscale_prior_concentration_high")),
            "lengthscale_prior_rate": Uniform(kwargs.get("lengthscale_prior_rate_low"), kwargs.get("lengthscale_prior_rate_high")),
        }       

        self.prior_args_keylist = ["dataset_size_low", "dataset_size_high", "x_domain_size", "x_dimensions"]
        self.prior_args = dict()

        for metaprior_param in self.prior_args_keylist:
            self.prior_args[metaprior_param] = kwargs.get(metaprior_param)
    
    def decode_sample(self, sample: Tensor) -> dict[str, Tensor]:
        sample_dict = {k: sample[..., ix] for ix, k in enumerate(self.metapriors_keylist)}

        for arg in self.prior_args_keylist:
            sample_dict[arg] = self.prior_args[arg]
        
        return sample_dict

    def encode_sample(self, decoded_sample: dict[str, Tensor]) -> Tensor:
        return torch.tensor([decoded_sample[k] for k in self.metapriors_keylist])
    
    def sample(self, sample_shape):
        sampled_values = []

        for metaprior in self.metapriors.values():
            sampled_values.append(
                metaprior.sample(sample_shape)
            )
            
        return torch.stack(sampled_values, dim=-1)


class GaussianProcessPrior(Distribution):
    arg_constraints = {
            "loc": constraints.real,
            "covariance_matrix": constraints.positive
        }

    def __init__(self, 
                 dataset_size_low: int,
                 dataset_size_high: int,
                 x_domain_size: float, 
                 x_dimensions: int,
                 lengthscale_prior_concentration: float,
                 lengthscale_prior_rate: float,
                 loc: Tensor = torch.zeros(torch.Size()), 
                 covariance_matrix: Tensor = torch.ones(torch.Size()),
        ):
        self.loc = loc
        self.covariance_matrix = covariance_matrix
        self.hyperparameter_batch_shape = self.loc.shape
        
        super().__init__()

        assert self.covariance_matrix.shape == self.loc.shape
        
        self.dataset_size_low = dataset_size_low
        self.dataset_size_high = dataset_size_high
        self.x_domain_size = x_domain_size
        self.lengthscale_prior = gpytorch.priors.GammaPrior(lengthscale_prior_concentration, lengthscale_prior_rate)

        self.mean_function = ConstantMean(batch_shape=self.hyperparameter_batch_shape, event_shape=torch.Size([1]))
        self.mean_function.constant = self.loc
        self.mean_function.to(self.loc.device)

        # Mean is constant and kernel is stationary; prior of y is same regardless of x
        self.x_distribution = Uniform(0, torch.Tensor([self.x_domain_size] * x_dimensions))

    @property
    def batch_shape(self):
        return torch.Size([self.hyperparameter_batch_shape[0]])

    @property
    def event_shape(self):
        return torch.Size([2])

    def get_kernel(self, lengthscale, sample_shape=torch.Size([])):
        batch_shape = sample_shape + self.hyperparameter_batch_shape
        kernel = ScaleKernel(RBFKernel(batch_shape=batch_shape, event_shape=torch.Size([1])), batch_shape=batch_shape, event_shape=torch.Size([1]))
        kernel.base_kernel.lengthscale = lengthscale
        kernel.outputscale = (self.covariance_matrix**0.5).view((1,)*len(sample_shape) + self.covariance_matrix.shape).expand(sample_shape + self.covariance_matrix.shape)
        kernel.to(self.covariance_matrix.device)
        
        return kernel

    def get_target_y_distribution(self, lengthscale, sample_shape=torch.Size([])):
        x = self.x_distribution.sample((1,)).to(self.loc.device)
        kernel = self.get_kernel(lengthscale, sample_shape)

        return MultivariateNormal(
            loc=self.mean_function(x).view((1,)*len(sample_shape) + self.hyperparameter_batch_shape + (1,)).expand(sample_shape + self.hyperparameter_batch_shape + (1,)),
            covariance_matrix=add_jitter(kernel(x).to_dense())
        )

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        lengthscale = self.lengthscale_prior.sample(sample_shape).unsqueeze(-1)
        y = self.get_target_y_distribution(lengthscale, sample_shape).sample()
        
        y_lengthscale = torch.cat([y, lengthscale], dim=-1)
        return y_lengthscale

    def log_prob(self, value: torch.Tensor):
        sample_shape = value.shape[:-(len(self.hyperparameter_batch_shape)+1)]
        log_prob_lengthscale = self.lengthscale_prior.log_prob(value[...,1])
        log_prob_y_given_lengthscale = self.get_target_y_distribution(value[...,[1]], sample_shape).log_prob(value[...,[0]])
        
        # log p(\theta, y) = log p(y|\theta) + log p(\theta)
        return log_prob_lengthscale + log_prob_y_given_lengthscale
            
    def sample_fulldataset(self) -> Tensor:

        dataset_size = torch.randint(
            self.dataset_size_low, 
            self.dataset_size_high, 
            [1]
        ).item()
        
        Dx = self.x_distribution.sample(self.hyperparameter_batch_shape + (dataset_size,))
        x = self.x_distribution.sample(self.hyperparameter_batch_shape +  (1,))

        lengthscale = self.lengthscale_prior.sample().unsqueeze(-1)
        y = self.get_target_y_distribution(lengthscale).sample()
        
        y_lengthscale = torch.cat([y, lengthscale], dim=-1)
        
        return Dx, x, y_lengthscale


class GPPredictiveObservationModel(ObservationModel):
    def __init__(self, observation_type:str):
        super().__init__()

        assert observation_type in ["dataset", "query"]

        self.observation_type = observation_type
    
    def condition_(self, Dx, x, y_lengthscale, gp_prior: Distribution):
        
        y = y_lengthscale[...,[0]]
        lengthscale = y_lengthscale[...,[1]]
        
        self.Dx = Dx
        self.x = x 
        self.y = y
        self.lengthscale = lengthscale

        self.hyperparameter_batch_shape = gp_prior.hyperparameter_batch_shape
        self.n_observations = Dx.shape[-1]
        
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = 0.001

        self.gp_posterior = ExactGPModel(
            self.x, self.y, self.likelihood, gp_prior.mean_function, gp_prior.get_kernel(lengthscale)
        )
        # We do not want to fit hypers, so we skip training
        self.gp_posterior.eval()
        self.conditional_distribution = self.gp_posterior(self.Dx)
    
    def sample(self, sample_shape = ...):
        if self.observation_type == "dataset":
            Dy = self.conditional_distribution.sample()
            return torch.cat(
                [
                    self.Dx, 
                    Dy.unsqueeze(-1)
                ],
                dim=-1
            )
        
        elif self.observation_type == "query":
            return self.x.squeeze(-2)
        
    def conditional_mean(self, x: Tensor):
        """
        Get mean of distribution conditioned on state. Also conditions self in place.
        Args:
            x: State on which to condition.

        Returns:
            Mean of distribution conditioned on x.

        """
        self.condition_(x)
        if self.observation_type == "dataset":
            return self.conditional_distribution.loc

        elif self.observation_type == "query":
            return self.x

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        device = value.device
        if self.observation_type == "dataset":
            return self.conditional_distribution.log_prob(value)

        elif self.observation_type == "query":
            return torch.zeros_like(value, device=device)


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
        prior = self.prior(**phi_decoded)
        Dx, x, y_lengthscale = prior.sample_fulldataset()
        for observation_model in self.observation_model.values():
            observation_model.condition_(Dx, x, y_lengthscale, prior)
        z = {key: observation_model.sample() for key, observation_model in self.observation_model.items()}

        return phi, y_lengthscale, z
    

def run(n_components: int,
        state_size: int,
        meta_prior_kwargs: dict,
        component_embedding_kwargs: dict,
        observation_embedding_kwargs: dict[str, dict],
        distribution_embedding_kwargs: dict,
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
    observation_embedding = {key: ObservationEmbedding(d_model=d_model, observation_size= (meta_prior_kwargs["x_dimensions"] + (1 if key=="dataset" else 0)), **kwargs)
                             for key, kwargs in observation_embedding_kwargs.items()}
    prior_embedding = HyperpriorEmbedding(d_model=d_model,n_components=n_components,state_size=state_size, **distribution_embedding_kwargs, **component_embedding_kwargs)
    model = DistributionTransformer(component_embedding=component_embedding,
                                    transformer_kwargs=transformer_kwargs,
                                    n_components=n_components,
                                    prior_embedding=prior_embedding,
                                    sample_space_transform=None,
                                    **observation_embedding)

    model, last_epoch_metrics = train(model, complete_distribution, _run=_run, **training_kwargs)

    test_gp(model, complete_distribution,
         bounds_func=partial(gmm_bounds_func,
                             scale_parametrisation=component_embedding_kwargs["scale_parametrisation"]),
         linspace_size=1000,
         hyperpior=True,
         _run=_run, **testing_kwargs)