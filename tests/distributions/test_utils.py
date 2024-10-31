import pytest
import torch
from torch.distributions import Distribution, Normal, MultivariateNormal, kl_divergence

from distributions.distributions import GaussianMixtureModel, LinearGaussianObservationModel
from distributions.utils import gmm_with_linear_gaussian_observations_posterior
from distributions.utils import kl_divergence as utils_kl_divergence


@pytest.mark.parametrize(
    ("prior", "observation_model", "observations", "expected_result"),
    [
        (
            GaussianMixtureModel(weights=torch.ones(1),
                                 loc=torch.zeros(1, 2),
                                 covariance_matrix=torch.eye(2).broadcast_to(1, 2, 2)),
            LinearGaussianObservationModel(observation_matrix=torch.eye(2),
                                           covariance_matrix=torch.eye(2)),
            torch.zeros(2),
            GaussianMixtureModel(weights=torch.ones(1),
                                 loc=torch.zeros(1, 2),
                                 covariance_matrix=0.5 * torch.eye(2).broadcast_to(1, 2, 2))
        ),
        (
            GaussianMixtureModel(weights=torch.tensor([1, 2])/3,
                                 loc=torch.tensor([[-1., -1.],
                                                   [1., 1.]]),
                                 covariance_matrix=torch.tensor([[[1., 0.5],
                                                                 [0.5, 2.]],
                                                                 [[1., -0.5],
                                                                 [-0.5, 1.]]])),
            LinearGaussianObservationModel(observation_matrix=torch.ones(1, 2),
                                           covariance_matrix=torch.eye(1)),
            torch.ones(1),
            GaussianMixtureModel(weights=torch.tensor([0.1417, 0.8583]),
                                 loc=torch.tensor([[-0.1, 0.5],
                                                   [0.75, 0.75]]),
                                 covariance_matrix=torch.tensor([[[0.55, -0.25],
                                                                  [-0.25, 0.75]],
                                                                 [[0.875, -0.625],
                                                                 [-0.625, 0.875]]])),
        ),
        (
            GaussianMixtureModel(weights=torch.ones(2, 1),
                                 loc=torch.zeros(2, 1, 2),
                                 covariance_matrix=torch.eye(2).broadcast_to(2, 1, 2, 2)),
            LinearGaussianObservationModel(observation_matrix=torch.eye(2),
                                           covariance_matrix=torch.eye(2)),
            torch.zeros(2, 2),
            GaussianMixtureModel(weights=torch.ones(2, 1),
                                 loc=torch.zeros(2, 1, 2),
                                 covariance_matrix=0.5 * torch.eye(2).broadcast_to(2, 1, 2, 2))
        )
    ]
)
def test_gmm_with_linear_gaussian_observations_posterior(prior: GaussianMixtureModel,
                                                         observation_model: LinearGaussianObservationModel,
                                                         observations: torch.Tensor,
                                                         expected_result: GaussianMixtureModel):
    posterior = gmm_with_linear_gaussian_observations_posterior(prior, observation_model, observations)
    print(posterior.weights)
    print(expected_result.weights)
    print(posterior.loc)
    print(expected_result.loc)
    print(posterior.covariance_matrix)
    print(expected_result.covariance_matrix)
    assert torch.allclose(posterior.weights, expected_result.weights, atol=1e-4)
    assert torch.allclose(posterior.loc, expected_result.loc, atol=1e-4)
    assert torch.allclose(posterior.covariance_matrix, expected_result.covariance_matrix, atol=1e-4)


@pytest.mark.parametrize(
    ("p", "q", "expected_result"),
    [
        (
            Normal(loc=0,
                   scale=1),
            Normal(loc=1,
                   scale=2),
            kl_divergence(Normal(loc=0,
                                 scale=1),
                          Normal(loc=1,
                                 scale=2))
        ),
        (
            MultivariateNormal(loc=torch.zeros(2),
                               covariance_matrix=torch.eye(2)),
            MultivariateNormal(loc=torch.ones(2),
                               covariance_matrix=2*torch.eye(2)),
            kl_divergence(MultivariateNormal(loc=torch.zeros(2),
                                             covariance_matrix=torch.eye(2)),
                          MultivariateNormal(loc=torch.ones(2),
                                             covariance_matrix=2*torch.eye(2)))
        ),
        (
            MultivariateNormal(loc=torch.zeros(2).broadcast_to(10, 2),
                               covariance_matrix=torch.eye(2).broadcast_to(10, 2, 2)),
            MultivariateNormal(loc=torch.ones(2).broadcast_to(10, 2),
                               covariance_matrix=2*torch.eye(2).broadcast_to(10, 2)),
            kl_divergence(MultivariateNormal(loc=torch.zeros(2),
                                             covariance_matrix=torch.eye(2)),
                          MultivariateNormal(loc=torch.ones(2),
                                             covariance_matrix=2*torch.eye(2))).broadcast_to(10)
        ),
    ]
)
def test_kl_divergence(p: Distribution, q: Distribution, expected_result: torch.Tensor):
    div = utils_kl_divergence(p, q)
    assert torch.allclose(div, expected_result, atol=0.01)
