"""
Experiment to test method with multiple nonlinear, but gaussian observations.
"""

import torch
from torch import Tensor

from distributions.distributions import (GaussianMixtureModelConjugateMetaPrior, MappedGaussianObservationModel,
                                         CompleteDistribution, GaussianMixtureModel)
from distributions.utils import plot_distributions, decode_gmm_sample
from model.embeddings import ComponentEmbedding, ObservationEmbedding
from model.distribution_transformer import DistributionTransformer
from workflows.train import train


def run(n_components: int,
        state_size: int,
        meta_prior_kwargs: dict,
        observation_covariance_matrix: dict[str, list[list[float]]],
        component_embedding_kwargs: dict,
        observation_embedding_kwargs: dict[str, dict],
        transformer_kwargs: dict,
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
        observation_covariance_matrix: Dictionary of covariance matrices for gaussian observations.
        component_embedding_kwargs: Dictionary of component embedding parameters.
        observation_embedding_kwargs: Dictionary of dictionaries of observation embedding parameters.
        transformer_kwargs: Dictionary of parameters for the transformer model.
        training_kwargs: Dictionary of parameters for the training routine.
        testing_kwargs: Dictionary of parameters for the testing routine.
        _run: Sacred run object.

    Returns:

    """

    # Meta-prior
    meta_prior = GaussianMixtureModelConjugateMetaPrior(state_size=state_size, n_components=n_components,
                                                        **meta_prior_kwargs)

    # Observation model
    mapping_dict = {
        "obs_1": lambda x: torch.sum(x ** 2, dim=-1).unsqueeze(-1),
        "obs_2": lambda x: torch.sinc(3*x)
    }
    covariance_matrix_dict = {key: torch.tensor(val, dtype=torch.float32)
                              for key, val in observation_covariance_matrix.items()}
    observation_model = {key: MappedGaussianObservationModel(covariance_matrix=covariance_matrix_dict[key],
                                                             mapping=mapping_dict[key])
                         for key in covariance_matrix_dict}

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

    scale_parametrisation = component_embedding_kwargs["scale_parametrisation"]

    with torch.no_grad():

        phi, x, z = complete_distribution.sample()
        phi_in, phi_out = model(phi, **z)
        exact_prior = GaussianMixtureModel(**decode_gmm_sample(phi, scale_parametrisation))
        prior = GaussianMixtureModel(**decode_gmm_sample(phi_in, scale_parametrisation))
        posterior = GaussianMixtureModel(**decode_gmm_sample(phi_out, scale_parametrisation))

        def bounds_func(phi: dict[str, Tensor]) -> tuple[float, float]:
            if scale_parametrisation == "precision_matrix":
                index = (phi[scale_parametrisation].flatten() / phi["weights"].flatten()).argmin()
                max_std = 1 / phi[scale_parametrisation][index].flatten().sqrt().item()
            else:
                index = (phi[scale_parametrisation].flatten() * phi["weights"].flatten()).argmax()
                max_std = phi[scale_parametrisation][index].flatten().sqrt().item()
            max_loc = phi["loc"].max().item()
            min_loc = phi["loc"].min().item()
            return min_loc - 4 * max_std, max_loc + 4 * max_std

        plot_distributions(exact_prior, prior, bounds=bounds_func(decode_gmm_sample(phi, scale_parametrisation)))
        plot_distributions(posterior, bounds=bounds_func(decode_gmm_sample(phi_out, scale_parametrisation)))
