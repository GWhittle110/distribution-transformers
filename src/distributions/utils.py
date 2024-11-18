"""
Utility functions for distributions
"""

import torch
from torch.distributions import Distribution, MultivariateNormal, Wishart
from torch.autograd.functional import jacobian
from torch.types import _size

from sklearn.mixture import GaussianMixture
from typing import Optional, Callable
from functools import partial

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
                  q_transform_shape_in: Optional[_size] = None,
                  q_transform_shape_out: Optional[_size] = None,
                  n_samples: int = 1000000):
    """
    Compute a stochastic approximation to KL[p||q]
    Args:
        p: Distribution p.
        q: Distribution q.
        q_transform: Transform from sample space of p to sample space of q.
            Defaults to None.
        q_transform_shape_in: Shape of input to q_transform.
        q_transform_shape_out: Shape of output to q_transform.
        n_samples: Number of samples with which to compute stochastic approximation.

    Returns:
        Stochastic approximation to KL[p||q].

    """
    samples = p.sample((n_samples,))
    q_samples = samples if q_transform is None else q_transform(samples)
    evaluations = p.log_prob(samples) - q.log_prob(q_samples)
    if q_transform is not None:
        assert q_transform_shape_in is not None, "Must specify input and output shapes of q_transform"
        n_in = torch.prod(torch.tensor(q_transform_shape_in)).to(torch.int).item()
        n_out = torch.prod(torch.tensor(q_transform_shape_out)).to(torch.int).item()
        assert n_in == n_out, "Only transformations which preserve the number of elements are supported."
        batch_shape = samples.shape[:-len(q_transform_shape_in)] if len(q_transform_shape_in) else samples.shape

        samples = samples.flatten(end_dim=len(batch_shape)-1)
        evaluations -= torch.stack([torch.logdet(jacobian(q_transform, sample, vectorize=True).reshape(n_out, n_in))
                                    for sample in samples]).reshape(batch_shape)
    return evaluations.nanmean(dim=0)


def distribution_to_gmm(p: Distribution, n_components: int,
                        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                        n_samples: int = 1000, scale_parametrisation: str = "covariance_matrix",
                        *args, **kwargs) -> torch.Tensor:
    """
    Approximate an arbitrary distribution p with a Gaussian mixture model with a specified number of components.
    Samples n_samples from p and fits the GMM using Expectation Maximisation.

    Examples:
        >>> root_state_size = 2
        >>> p = Wishart(root_state_size + 1, torch.eye(root_state_size))
        >>> state_size = root_state_size ** 2
        >>> n_components = 4
        >>> transform = partial(torch.flatten, start_dim=-2)
        >>> phi = distribution_to_gmm(p, n_components, transform)
        >>> weights = phi[..., :n_components]
        >>> loc = phi[..., n_components:(1 + state_size) * n_components].unflatten(-1, (n_components, state_size))
        >>> scale = phi[..., -n_components * state_size ** 2:].unflatten(-1, (n_components, state_size, state_size))
        >>> gmm = GaussianMixtureModel(weights, loc, covariance_matrix=scale)
        >>> print(kl_divergence(p, gmm, transform))

    Args:
        p: Probability distribution to approximate.
        n_components: Number of components in approximating GMM.
        transform: Transform under which to fit the GMM.
            Defaults to None.
        n_samples: Number of samples on which to perform EM.
            Defaults to 1000.
        scale_parametrisation: Parametrisation of scale parameter of GMM. One of "covariance_matrix", "precision_matrix"
            or "scale_tril".
            Defaults to "covariance_matrix".
        *args, **kwargs for sklearn GaussianMixture.

    Returns:
        Tensor of distribution parameters

    """
    samples = p.sample((n_samples,))
    if transform is not None:
        samples = transform(samples)
    device = samples.device
    samples = samples.cpu().detach()
    if samples.dim() > 2:
        batch_shape = samples.shape[1:-1]
        samples = torch.movedim(samples, 0, -2)
    else:
        samples = samples.unsqueeze(0)
        batch_shape = tuple()

    # Inner function for handling batched distributions
    def inner(sample):
        gmm = GaussianMixture(n_components=n_components, *args, **kwargs)
        gmm.fit(sample)
        weights = torch.tensor(gmm.weights_)
        loc = torch.tensor(gmm.means_)
        match scale_parametrisation:
            case "covariance_matrix":
                scale = torch.tensor(gmm.covariances_)
            case "precision_matrix":
                scale = torch.tensor(gmm.precisions_)
            case "scale_tril":
                scale = torch.cholesky(torch.tensor(gmm.covariances_))
            case _:
                raise ValueError('scale_parametrisation must be one of "covariance_matrix", "precision_matrix" or '
                                 '"scale_tril')
        return torch.hstack([weights.flatten(), loc.flatten(), scale.flatten()])

    phi = torch.cat([inner(sample) for sample in samples.flatten(0, -3)]).to(device)
    return phi.reshape(batch_shape + (-1,))
