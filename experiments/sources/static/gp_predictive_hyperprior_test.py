"""
Experiment evaluating method on problem of finding posterior for GP hyperparameters
"""

from copy import copy, deepcopy
from functools import partial
from random import sample
from typing import Union, Sequence, Callable, Optional
import gpytorch
from matplotlib import pyplot as plt
import test
from torch.func import vmap, jacrev
from sympy import comp, hyper
import torch
from torch import Tensor
from torch import nn
from torch.distributions import Normal, MultivariateNormal, constraints, Distribution, Uniform, Normal, Independent
from torch.distributions.utils import lazy_property
from torch.types import _size
from competitor_methods.pfns import RiemannDistribution, PFN, get_borders_from_prior
from competitor_methods.variational_inference import VI
from workflows.train import train_pfn
from distributions.utils import decode_gmm_sample

from gpytorch import add_jitter
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from time import time

from distributions.distributions import (InverseGammaMetaPrior, ObservationModel, CompleteDistribution,
                                         GaussianMixtureModel, MetaPrior)
from model.embeddings import DistributionEmbedding
from model.distribution_transformer import DistributionTransformer
from distributions.utils import gmm_bounds_func
from workflows.train import train
from workflows.test import test_gp
from model.embeddings import ComponentEmbedding, GammaEmbedding, ObservationEmbedding
from experiments.sources.static.gp_predictive_hyperprior import CompleteDistributionGPPredictive, ExactGPModel, GPPredictiveObservationModel, HyperpriorEmbedding, MeanScaleMetaPrior
    

    
def test_vi(
    n_test_priors,
    model,
    meta_prior_kwargs,
    observation_model,
    device,
    competitor_kwargs,
    mc_samples = 100
):
    
    meta_prior_vi = MeanScaleMetaPrior(marginalise_y=True, **meta_prior_kwargs)
    complete_distribution_vi = CompleteDistributionGPPredictive(meta_prior_vi, marginalise_y=True, **observation_model)

    phi, x, z = complete_distribution_vi.sample((n_test_priors,))
    
    phi_prior_dict = complete_distribution_vi.meta_prior.decode_sample(phi)
    prior = complete_distribution_vi.meta_prior.prior(**phi_prior_dict)
    
    vi = VI(1, prior, complete_distribution_vi.observation_model, torch.exp,
                    **competitor_kwargs["vi"]).to(device)
    
    before_vi = time()
    torch.set_grad_enabled(True)
    vi.fit(z, **competitor_kwargs["vi"])
    torch.set_grad_enabled(False)
    vi_time = time() - before_vi
    
    vi_nll = -vi.distribution().log_prob(torch.log(x)
                                            .reshape(phi.shape[:1]
                                                    +torch.Size([1])).to(device))
    vi_nll -= torch.logdet(vmap(jacrev(torch.log))
                            (x.reshape(phi.shape[:1]
                                        +torch.Size([1])).to(device)
                            ).reshape(prior.batch_shape +torch.Size([1])
                                        +torch.Size([1])))
    vi_expected_nll = vi_nll.mean().item()
    vi_std_nll = vi_nll.std().item()
    
    print(f"lengthscale: Testing VI NNL {vi_expected_nll} +/- {1.96 * vi_std_nll/ (phi.shape[0]) ** 0.5} Time taken {vi_time}")
    
def test_pfn(
    n_test_priors,
    model,
    meta_prior_kwargs,
    observation_model,
    pfn,
    marginalise_lengthscale=False,
    marginalise_y=False
    ):

    meta_prior_pfn = MeanScaleMetaPrior(marginalise_lengthscale=marginalise_lengthscale, marginalise_y=marginalise_y, **meta_prior_kwargs)
    complete_distribution_pfn = CompleteDistributionGPPredictive(meta_prior_pfn, marginalise_lengthscale=marginalise_lengthscale, marginalise_y=marginalise_y, **observation_model)
    phi_buckets, x_buckets = complete_distribution_pfn.sample((n_test_priors, 1))[:2]
    
    borders = get_borders_from_prior(complete_distribution_pfn.meta_prior.prior(
                        **complete_distribution_pfn.meta_prior.decode_sample(phi_buckets)), pfn.n_buckets, pfn.infinite_support,
                        pfn.leftmost_border, pfn.rightmost_border).mean(dim=0)
    pfn.borders = borders
    
    _, x, z = complete_distribution_pfn.sample((n_test_priors,))
    pfn = pfn.cpu()
    before_inference = time()
    phi_out = pfn(**z)
    after_inference = time() - before_inference
    pfn_posterior = RiemannDistribution(phi_out, pfn.borders, pfn.infinite_support)
    
    posterior_losses = -pfn_posterior.log_prob(x.reshape(pfn_posterior.batch_shape))

    variable = "y" if marginalise_lengthscale else "legthscale"

    print(f"{variable}: Testing PFN NNL {posterior_losses.mean().item()} +/- {1.96 * posterior_losses.std().item() / (pfn_posterior.batch_shape[0]) ** 0.5} Time taken: {after_inference} seconds")

    return {
        "expected_nnl": posterior_losses.mean().item(),
        "vi_std_nll": posterior_losses.std().item()
    }
    
def test_dt_individual(
    n_test_priors,
    model,
    meta_prior_kwargs,
    observation_model,
    variable
    ):

    device = "cuda:0" if next(model.parameters()).is_cuda else "cpu"
    
    meta_prior = MeanScaleMetaPrior(**meta_prior_kwargs,marginalise_lengthscale=variable=="y", marginalise_y=not(variable=="y"))
    complete_distribution = CompleteDistributionGPPredictive(meta_prior,  **observation_model, marginalise_lengthscale=variable=="y", marginalise_y=not(variable=="y"))

    phi, x, z = complete_distribution.sample((n_test_priors,))
    
    before_inference = time()
    _, phi_out = model(phi, **z)
    after_inference = time() - before_inference
    
    scale_parametrisation = model.component_embedding.scale_parametrisation
    
    phi_prior_dict = complete_distribution.meta_prior.decode_sample(phi)
    prior = complete_distribution.meta_prior.prior(**phi_prior_dict)

    model_posterior = GaussianMixtureModel(**decode_gmm_sample(phi_out, scale_parametrisation))
    posterior_losses = -model_posterior.log_prob(model.sample_space_transform(x))

    posterior_losses -= torch.logdet(vmap(jacrev(model.sample_space_transform))
                                        (x.reshape(model_posterior.batch_shape
                                                    + model_posterior.event_shape).to(device)
                                            ).reshape(prior.batch_shape + model_posterior.event_shape
                                                    + model_posterior.event_shape))

    print(f"{variable}: Testing DT NNL {posterior_losses.mean().item()} +/- {1.96 * posterior_losses.std().item() / (model_posterior.batch_shape[0]) ** 0.5} Time taken: {after_inference}")
    
def get_model_size(model):
    param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
    buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
    return (param_size + buffer_size) / 1024 ** 2   # Return size in MB
    
    
def test_dt(
    n_test_priors,
    model,
    meta_prior_kwargs,
    observation_model,
    ):

    device = "cuda:0" if next(model.parameters()).is_cuda else "cpu"
    
    meta_prior = MeanScaleMetaPrior(**meta_prior_kwargs)
    complete_distribution = CompleteDistributionGPPredictive(meta_prior,  **observation_model)

    phi, x, z = complete_distribution.sample((n_test_priors,))
    
    before_inference = time()
    _, phi_out = model(phi, **z)
    after_inference = time() - before_inference
    
    scale_parametrisation = model.component_embedding.scale_parametrisation
    
    phi_prior_dict = complete_distribution.meta_prior.decode_sample(phi)
    prior = complete_distribution.meta_prior.prior(**phi_prior_dict)
    
    model_posterior = GaussianMixtureModel(**decode_gmm_sample(phi_out, scale_parametrisation))
    posterior_losses = - model_posterior.log_prob(model.sample_space_transform(x))
    posterior_losses -= torch.logdet(vmap(jacrev(model.sample_space_transform))
                                            (x.reshape(model_posterior.batch_shape
                                                       + torch.Size([2])).to(device)
                                             ).reshape(prior.batch_shape + torch.Size([2, 2])))
    
    print(f"joint: Testing DT NNL {posterior_losses.mean().item()} +/- {1.96 * posterior_losses.std().item() / (model_posterior.batch_shape[0]) ** 0.5} Time taken: {after_inference}")

    for variable in ["y",  "lengthscale"]:
        var_ix = 0 if variable=="y" else 1
        model_posterior = get_marginal_posterior(phi_out, scale_parametrisation, marginalise_lengthscale=variable=="y", marginalise_y=not(variable=="y"))
        posterior_losses = -model_posterior.log_prob(model.sample_space_transform(x)[...,[var_ix]])

        posterior_losses -= torch.logdet(vmap(jacrev(model.sample_space_transform))
                                            (x.reshape(model_posterior.batch_shape
                                                       + torch.Size([2])).to(device)
                                             )[...,var_ix, var_ix].reshape(prior.batch_shape + model_posterior.event_shape
                                                       + model_posterior.event_shape))

        print(f"{variable}: Testing DT NNL {posterior_losses.mean().item()} +/- {1.96 * posterior_losses.std().item() / (model_posterior.batch_shape[0]) ** 0.5} Time taken: {after_inference}")


def get_marginal_posterior(
    phi_out,
    scale_parametrisation,
    marginalise_y=False,
    marginalise_lengthscale=False,
):
    
    assert marginalise_y or marginalise_lengthscale
    assert not(marginalise_y and marginalise_lengthscale)
    
    var_ix = 0 if marginalise_lengthscale else 1
    
    weigth = phi_out[...,0]
    loc = phi_out[..., 1: 3]
    scale = phi_out[..., -2 ** 2:].reshape(*phi_out.shape[:-1], 2, 2)
    var = torch.diagonal(scale, dim1=-2, dim2=-1)
    marginal_phi_out = torch.stack([weigth, loc[:,:,var_ix], var[:,:,var_ix]], dim=-1)

    return GaussianMixtureModel(**decode_gmm_sample(marginal_phi_out, scale_parametrisation))
    

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
    
    with torch.no_grad():
        device = 'cpu:0'
        competitor_kwargs = testing_kwargs["competitor_kwargs"]
        pfn_kwargs = copy(competitor_kwargs["pfns"])
        del pfn_kwargs["training_kwargs"]
        n_test_priors = testing_kwargs.get("n_test_priors")

        observation_model = {observation_type: GPPredictiveObservationModel(observation_type=observation_type)
                            for observation_type in ["dataset", "query"]}

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
                                        sample_space_transform= lambda x: torch.stack([x[...,0], torch.log(x[...,1])], dim=-1),
                                        **observation_embedding)
        
        if kwargs.get("model_path", False):
            model.load_state_dict(torch.load(kwargs.get("model_path"), weights_only=True))
            print(f"DF model size {get_model_size(model)}")
            test_dt(n_test_priors, model, meta_prior_kwargs, observation_model)
        else:
            model.load_state_dict(torch.load(kwargs.get("model_y_path"), weights_only=True))
            test_dt_individual(n_test_priors, model, meta_prior_kwargs, observation_model, "y")
            model.load_state_dict(torch.load(kwargs.get("model_ls_path"), weights_only=True))
            test_dt_individual(n_test_priors, model, meta_prior_kwargs, observation_model, "ls")
        
        pfn_y = PFN(**pfn_kwargs, **deepcopy(model.observation_embeddings))
        pfn_y.load_state_dict(torch.load(kwargs.get("pfn_y_path"), weights_only=True))
        print(f"pfn_y model size {get_model_size(pfn_y)}")
        
        # PFNs
        pfn_ls = PFN(**pfn_kwargs, **deepcopy(model.observation_embeddings))
        pfn_ls.load_state_dict(torch.load(kwargs.get("pfn_ls_path"), weights_only=True))
        print(f"pfn_ls model size {get_model_size(pfn_ls)}")

        
        test_pfn(n_test_priors, model, meta_prior_kwargs, observation_model, pfn_y ,marginalise_lengthscale=True)
        test_pfn(n_test_priors, model, meta_prior_kwargs, observation_model, pfn_ls ,marginalise_y=True)
        test_vi(n_test_priors, model, meta_prior_kwargs, observation_model, device, competitor_kwargs)
        