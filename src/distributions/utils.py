"""
Utility functions for distributions
"""

import torch
from torch.distributions import Distribution, MultivariateNormal, InverseGamma, Normal
from torch.func import vmap, jacrev
from typing import Optional

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
    schur_marginal_term = (schur_marginal_term.to(torch.float64) +
                           torch.transpose(schur_marginal_term.to(torch.float64), dim0=-2, dim1=-1)) / 2
    schur_marginal_term = schur_marginal_term.to(torch.float32)
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
    posterior_covariance_matrix = posterior_covariance_matrix.to(torch.float32)

    observation_component_marginals = MultivariateNormal(loc=observation_marginal_mean,
                                                         covariance_matrix=schur_marginal_term)
    observation_component_evidences = torch.exp(observation_component_marginals.log_prob(observations))

    posterior_weights = prior.weights * observation_component_evidences
    posterior_weights /= posterior_weights.sum(dim=-1).unsqueeze(-1)

    posterior = GaussianMixtureModel(weights=posterior_weights, loc=posterior_loc,
                                     covariance_matrix=posterior_covariance_matrix, validate_args=False)
    return posterior


def kl_divergence(p: Distribution, q: Distribution,
                  q_transform: Optional[callable] = None,
                  n_samples: int = 100000):
    """
    Compute a stochastic approximation to KL[p||q]

    Example - transform, no batching:
        >>> p = InverseGamma(1, 1)
        >>> q = Normal(0, 1)
        >>> q_transform = torch.log
        >>> print(kl_divergence(p, q, q_transform))

    Example - transform, batching:
        >>> p = InverseGamma(torch.ones(2, 2), torch.ones(2, 2))
        >>> q = Normal(torch.zeros(2, 2), torch.ones(2, 2))
        >>> q_transform = torch.log
        >>> print(kl_divergence(p, q, q_transform))

    Args:
        p: Distribution p.
        q: Distribution q.
        q_transform: Transform from sample space of p to sample space of q.
            Defaults to None.
        n_samples: Number of samples with which to compute stochastic approximation.

    Returns:
        Stochastic approximation to KL[p||q].

    """
    samples = p.sample((n_samples,))
    q_samples = samples if q_transform is None else q_transform(samples)
    evaluations = p.log_prob(samples) - q.log_prob(q_samples)
    evaluations[evaluations == -float("inf")] = torch.nan
    if q_transform is not None:
        n_in = torch.prod(torch.tensor(p.event_shape)).to(torch.int).item()
        n_out = torch.prod(torch.tensor(q.event_shape)).to(torch.int).item()
        assert n_in == n_out, "Only transformations which preserve the number of elements are supported."
        if len(p.batch_shape):
            samples = samples.flatten(end_dim=len(p.batch_shape))
        evaluations -= torch.logdet(vmap(jacrev(q_transform))(samples).reshape((-1,) + p.batch_shape + (n_out, n_in)))
    return evaluations.nanmean(dim=0)
