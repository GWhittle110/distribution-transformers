"""
Utility functions for distributions
"""

import torch
from torch.distributions import Distribution, MultivariateNormal

from distributions.distributions import GaussianMixtureModel, LinearGaussianObservationModel


def gmm_with_linear_gaussian_observations_posterior(prior: GaussianMixtureModel,
                                                    observation_model: LinearGaussianObservationModel,
                                                    observations: torch.Tensor
                                                    ) -> GaussianMixtureModel:
    """
    Given a Gaussian mixture model prior, a linear Gaussian observation model, and a set of observations, return the
    analytical posterior; another Gaussian mixture model.

    Args:
        prior: Prior Gaussian mixture model.
        observation_model: Linear Gaussian observation model.
        observations: Tensor of observations.

    Returns:
        Posterior Gaussian mixture model.

    """
    match prior.scale_parametrisation:
        case "covariance_matrix":
            prior_covariance_matrix = prior.covariance_matrix
        case "precision_matrix":
            prior_covariance_matrix = torch.linalg.inv(prior.precision_matrix)
        case "scale_tril":
            prior_covariance_matrix = torch.einsum("...ij,...kj->...ik", prior.scale_tril, prior.scale_tril)
        case _:
            raise ValueError

    match observation_model.scale_parametrisation:
        case "covariance_matrix":
            observation_covariance_matrix = observation_model.covariance_matrix
        case "precision_matrix":
            observation_covariance_matrix = torch.linalg.inv(observation_model.precision_matrix)
        case "scale_tril":
            observation_covariance_matrix = torch.einsum("...ij,...kj->...ik", observation_model.scale_tril,
                                                         observation_model.scale_tril)
        case _:
            raise ValueError

    observation_matrix = observation_model.observation_matrix
    
    schur_marginal_term = torch.einsum("ij,...jk,lk->...il", observation_matrix, prior_covariance_matrix,
                                       observation_matrix) + observation_covariance_matrix
    schur_inverse_term = torch.linalg.inv(schur_marginal_term)
    schur_covariance_term = torch.einsum("...ij,kj->...ik", prior_covariance_matrix, observation_matrix)
    observation_marginal_mean = torch.einsum("ij,...j->...i", observation_matrix, prior.loc)
    observations = observations.unsqueeze(-2)
    residual_term = observations - observation_marginal_mean

    posterior_loc = prior.loc + torch.einsum("...ij,...jk,...k->...i", schur_covariance_term, schur_inverse_term,
                                             residual_term)

    posterior_covariance_matrix = prior_covariance_matrix - torch.einsum("...ij,...jk,...lk->...il",
                                                                         schur_covariance_term, schur_inverse_term,
                                                                         schur_covariance_term)
    posterior_covariance_matrix = (posterior_covariance_matrix.to(torch.float64) +
                                   torch.transpose(posterior_covariance_matrix.to(torch.float64), dim0=-2, dim1=-1)) / 2

    observation_component_marginals = MultivariateNormal(loc=observation_marginal_mean,
                                                         covariance_matrix=schur_marginal_term)
    observation_component_evidences = torch.exp(observation_component_marginals.log_prob(observations))

    posterior_weights = prior.weights * observation_component_evidences
    posterior_weights /= posterior_weights.sum(dim=-1).unsqueeze(-1)

    posterior = GaussianMixtureModel(weights=posterior_weights, loc=posterior_loc,
                                     covariance_matrix=posterior_covariance_matrix)
    return posterior


def kl_divergence(p: Distribution, q: Distribution, n_samples: int = 1000000):
    samples = p.sample((n_samples,))
    evaluations = p.log_prob(samples) - q.log_prob(samples)
    return evaluations.nanmean(dim=0)
