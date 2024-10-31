import torch

from distributions.distributions import (GaussianMixtureModelConjugateMetaPrior, MappedGaussianObservationModel,
                                         CompleteDistribution)
from model.transformers import GMMTransformerModel
from model.train import train


n_components = 10
state_size = 2


def mapping(x: torch.Tensor) -> torch.Tensor:
    z1 = torch.einsum("...i,...i->...", x, x)
    z1 = z1.reshape(z1.shape + (1,))
    z2 = torch.sum(x, dim=-1)
    z2 = z2.reshape(z2.shape + (1,))
    return torch.cat([z2], dim=-1)


conjugate_meta_prior = GaussianMixtureModelConjugateMetaPrior(n_components=n_components, state_size=state_size,
                                                              scale_parametrisation="precision_matrix")
observation_model = MappedGaussianObservationModel(covariance_matrix=0.1 * torch.eye(1), mapping=mapping)
complete_distribution = CompleteDistribution(conjugate_meta_prior, observation_model)

model = GMMTransformerModel(n_components=n_components, state_size=state_size,
                            n_observations=observation_model.n_observations, d_model=16, n_head=8,
                            scale_parametrisation="precision_matrix")

model = train(model, complete_distribution, compute_prior_loss=True, lr=0.0001, warmup_epochs=5,
              progress_bar=True, verbose=True)
