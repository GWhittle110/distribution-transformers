"""
Experiment to validate method against closed form posterior of inverse prior with linear Gaussian observations

Procedure:
Sample ~10 observation models, state sizes and number of components from some distributions. For each:
    1) Train the variational transformer
    2) Sample ~1000 priors and observations as in training and compute posterior using variational transformer and
        analytical solution.
Compute average KL divergence KL[q || p]
"""

import torch
from torch import Tensor

from distributions.distributions import InverseGammaMetaPrior, MappedScaleGaussianObservationModel, CompleteDistribution
from model.distribution_transformer import DistributionTransformer
from workflows.train import train
from workflows.test import test_conjugate_prior
from model.embeddings import ComponentEmbedding, GammaEmbedding, ObservationEmbedding


def run(n_components: int,
        meta_prior_kwargs: dict,
        observation_loc: dict[str, list[float]],
        distribution_embedding_kwargs: dict,
        component_embedding_kwargs: dict,
        observation_embedding_kwargs: dict[str, dict],
        transformer_kwargs: dict,
        training_kwargs: dict,
        testing_kwargs: dict,
        _run=None,
        *args, **kwargs) -> None:
    """
    Run an experiment comparing distribution transformers to the closed form posterior of a GMM prior under linear
    Gaussian observations.

    Args:
        n_components: Number of GMM components.
        meta_prior_kwargs: Dictionary of parameters for the meta prior.
        observation_loc: Dictionary of means of observation distributions.
        distribution_embedding_kwargs: Dictionary of distribution embedding parameters.
        component_embedding_kwargs: Dictionary of component embedding parameters.
        observation_embedding_kwargs: Dictionary of dictionaries of observation embedding parameters.
        transformer_kwargs: Dictionary of parameters for the transformer model.
        training_kwargs: Dictionary of parameters for the training routine.
        testing_kwargs: Dictionary of parameters for the testing routine.
        _run: Sacred run object.

    """

    meta_prior = InverseGammaMetaPrior(**meta_prior_kwargs)

    observation_model = {key: MappedScaleGaussianObservationModel(torch.tensor(loc, dtype=torch.float32),
                                                                  scale_parametrisation="covariance_matrix",
                                                                  mapping=None)
                         for key, loc in observation_loc.items()}

    complete_distribution = CompleteDistribution(meta_prior, **observation_model)

    d_model = transformer_kwargs["d_model"]
    distribution_embedding = GammaEmbedding(d_model=d_model, n_components=n_components, **distribution_embedding_kwargs)
    component_embedding = ComponentEmbedding(state_size=1, d_model=d_model, **component_embedding_kwargs)
    observation_embedding = {key: ObservationEmbedding(d_model=d_model, observation_size=1, **kwargs)
                             for key, kwargs in observation_embedding_kwargs.items()}
    model = DistributionTransformer(component_embedding=component_embedding,
                                    transformer_kwargs=transformer_kwargs,
                                    n_components=n_components,
                                    prior_embedding=distribution_embedding,
                                    sample_space_transform=torch.log,
                                    **observation_embedding)

    model, last_epoch_metrics = train(model, complete_distribution, _run=_run, **training_kwargs)

    def conjugacy_update(phi: dict[str, Tensor],
                         z: dict[str, Tensor],
                         device: str
                         ) -> dict[str, Tensor]:
        return {
            "concentration": phi["concentration"] + len(z) / 2,
            "rate": phi["rate"] + sum((z[key].squeeze() - torch.tensor(observation_loc[key]).to(device).squeeze()) ** 2
                                      for key in z) / 2
        }

    def bounds_func(params: dict[str, Tensor]) -> tuple[float, float]:
        concentration = params["concentration"].item()
        rate = params["rate"].item()
        return 1e-6, 4 * rate / concentration + 1 / rate

    test_conjugate_prior(model, complete_distribution, conjugacy_update, bounds_func=bounds_func,
                         _run=_run, **testing_kwargs)
