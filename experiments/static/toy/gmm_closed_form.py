"""
Experiment to validate method against closed form posterior of GMM prior with linear Gaussian observations

Procedure:
Sample ~10 observation models, state sizes and number of components from some distributions. For each:
    1) Train the variational transformer
    2) Sample ~1000 priors and observations as in training and compute posterior using variational transformer and
        analytical solution.
Compute average KL divergence KL[q || p]
"""

import torch
from time import time

from distributions.distributions import (GaussianMixtureModelConjugateMetaPrior, LinearGaussianObservationModel,
                                         CompleteDistribution, GaussianMixtureModel)
from distributions.utils import gmm_with_linear_gaussian_observations_posterior, kl_divergence
from model.transformers import GMMTransformerModel, GMMConditionalTransformerModel
from model.train import train


n_components = 1
state_size = 1
observation_size = 1

n_test_priors = 1000
device = torch.device("cuda:0")


# Meta-prior
meta_prior = GaussianMixtureModelConjugateMetaPrior(state_size=state_size, n_components=n_components,
                                                    scale_parametrisation="covariance_matrix")

# Observation model
covariance_matrix = torch.eye(observation_size)
observation_matrix = torch.rand((observation_size, state_size))
observation_model = LinearGaussianObservationModel(observation_matrix=observation_matrix,
                                                   covariance_matrix=covariance_matrix)

# Complete distribution
complete_distribution = CompleteDistribution(meta_prior, observation_model)

# Variational transformer
model = GMMConditionalTransformerModel(n_components=n_components, state_size=state_size,
                                       n_observations=observation_model.n_observations, d_model=64, n_head=8,
                                       dim_feedforward=2048,
                                       scale_parametrisation="covariance_matrix")

model = train(model, complete_distribution, compute_prior_loss=True, warmup_epochs=10,
              epochs=50, progress_bar=True, verbose=True, lr=0.0003, batch_size=5000)
model.to(device)

with torch.no_grad():
    # Test inputs
    test_samples = complete_distribution.sample((n_test_priors,)).to(device)
    test_prior_params = complete_distribution.decode_sample(test_samples)["phi"]
    test_observations = complete_distribution.decode_sample(test_samples)["z"]
    test_priors = GaussianMixtureModel(**meta_prior.decode_sample(test_prior_params))

    # Exact solution
    observation_model.observation_matrix = observation_model.observation_matrix.to(device)
    observation_model.covariance_matrix = observation_model.covariance_matrix.to(device)
    exact_posterior = gmm_with_linear_gaussian_observations_posterior(test_priors, observation_model,
                                                                      test_observations)
    # Inference solution
    start_time = time()
    model_posterior_params = model(torch.cat([test_prior_params, test_observations], dim=-1))
    model_posterior = GaussianMixtureModel(**meta_prior.decode_sample(model_posterior_params))
    inference_time = time() - start_time
    average_inference_time = inference_time / n_test_priors

    kl_divergences = kl_divergence(model_posterior, exact_posterior, 10000)
    prior_kl_divergences = kl_divergence(test_priors, exact_posterior, 10000)
    print(kl_divergences.mean(), prior_kl_divergences.mean())
