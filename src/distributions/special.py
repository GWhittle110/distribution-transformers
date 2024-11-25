"""
Special methods and classes operating on distributions
"""

import torch
from torch.distributions import Distribution, Wishart
from torch.types import _size
from typing import Optional, Callable
from sklearn.mixture import GaussianMixture
from functools import partial

from distributions.distributions import (GaussianMixtureModel, MetaPrior, InverseGammaMetaPrior, ObservationModel,
                                         CompleteDistribution)
from distributions.utils import kl_divergence


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
    samples = torch.movedim(samples, 0, len(p.batch_shape))
    if len(p.batch_shape):
        samples = samples.flatten(end_dim=len(p.batch_shape)-1)
    else:
        samples = samples.unsqueeze(0)

    # Inner function for handling batched distributions
    def inner(sample):
        gmm = GaussianMixture(n_components=n_components, *args, **kwargs)
        gmm.fit(sample.reshape(n_samples, -1))
        weights = torch.tensor(gmm.weights_, dtype=torch.float32)
        loc = torch.tensor(gmm.means_, dtype=torch.float32)
        match scale_parametrisation:
            case "covariance_matrix":
                scale = torch.tensor(gmm.covariances_, dtype=torch.float32)
            case "precision_matrix":
                scale = torch.tensor(gmm.precisions_, dtype=torch.float32)
            case "scale_tril":
                scale = torch.cholesky(torch.tensor(gmm.covariances_, dtype=torch.float32))
            case _:
                raise ValueError('scale_parametrisation must be one of "covariance_matrix", "precision_matrix" or '
                                 '"scale_tril')
        return torch.hstack([weights.flatten(), loc.flatten(), scale.flatten()])

    phi = torch.cat([inner(sample) for sample in samples]).to(device)
    return phi.reshape(p.batch_shape + (-1,))


class ApproximateWarpedGMMMetaPrior(MetaPrior):
    arg_constraints = {}

    def __init__(self, meta_prior: MetaPrior, n_components: int,
                 transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 scale_parametrisation: str = "covariance_matrix",
                 cache_samples: bool = True):
        """
        Wrapper around a MetaPrior instance which approximates sampled priors with GMMs.

        Example:
            >>> meta_prior = InverseGammaMetaPrior()
            >>> wrapped_meta_prior = ApproximateWarpedGMMMetaPrior(meta_prior, 4, transform=torch.log)
            >>> print(wrapped_meta_prior.sample((10, 10)))
            >>> print(wrapped_meta_prior.exact_prior_samples)
            >>> p = meta_prior.prior(**meta_prior.decode_sample(wrapped_meta_prior.exact_prior_samples))
            >>> q = GaussianMixtureModel(**wrapped_meta_prior.decode_sample(wrapped_meta_prior.approximate_prior_samples))
            >>> print(kl_divergence(p, q, lambda x: torch.log(x.unsqueeze(-1))))

        Args:
            meta_prior: Wrapped MetaPrior instance.
            n_components: Number of components in approximating GMM.
            transform: Transform from meta_prior's sample space to GMM sample space.
                Defaults to None.
            scale_parametrisation: Parametrisation of GMM scale parameter. Must be one of "covariance_matrix",
                "precision_matrix" or "scale_tril".
                Defaults to "covariance_matrix".
            cache_samples: Whether to store sampled priors after first sampling. Useful for training.
                Defaults to True.

        """
        super().__init__(GaussianMixtureModel)
        self.meta_prior = meta_prior
        self.n_components = n_components
        self.transform = transform
        self.scale_parametrisation = scale_parametrisation
        self.cache_prior = cache_samples

        self.approximate_prior_samples: Optional[torch.Tensor] = None
        self.exact_prior_samples: Optional[torch.Tensor] = None

        assert scale_parametrisation in {"covariance_matrix", "precision_matrix", "scale_tril"}, \
            'scale_parametrisation must be one of "covariance_matrix", "precision_matrix", or "scale_tril".'

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        if (not self.cache_prior or self.approximate_prior_samples is None
                or self.approximate_prior_samples.shape[:-1] != sample_shape):
            prior = self.meta_prior.sample(sample_shape)
            if prior.shape == sample_shape:
                prior.unsqueeze(-1)
            decoded_prior = self.meta_prior.decode_sample(prior)
            approximate_prior = distribution_to_gmm(self.meta_prior.prior(**decoded_prior),
                                                    n_components=self.n_components,
                                                    transform=self.transform)

            self.exact_prior_samples = prior
            self.approximate_prior_samples = approximate_prior
            self.prior_size = self.approximate_prior_samples.shape[-1]

        return self.approximate_prior_samples

    def decode_sample(self, sample: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Decode tensor of sampled parameters to dictionary of tensors keyed by GMM parameter.

        Args:
            sample: Sampled tensor.

        Returns:
            Decoded sample.
        """
        sample_shape = sample.shape[:-1]
        weights = sample[..., :self.n_components]
        loc = sample[..., self.n_components:self.n_components * 2].reshape(*sample_shape, self.n_components, 1)
        scale = sample[..., -self.n_components:].reshape(*sample_shape, self.n_components, 1, 1)
        return {"weights": weights,
                "loc": loc,
                "covariance_matrix": scale}

    def encode_sample(self, decoded_sample: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode dictionary of sampled parameters to a singular tensor. Inverse operation of decode_sample.

        Args:
            decoded_sample:  Dictionary of decoded sample.

        Returns:
            Tensor encoding sample.
        """
        weights = decoded_sample["weights"]
        loc = decoded_sample["loc"]
        scale = decoded_sample["covariance_matrix"]
        return torch.cat([weights, loc.flatten(start_dim=-2), scale.flatten(start_dim=-3)], dim=-1)


class ApproximateCompleteDistribution(CompleteDistribution):
    has_rsample = True
    _validate_args = False

    def __init__(self, meta_prior: ApproximateWarpedGMMMetaPrior, observation_model: ObservationModel,
                 sample_mode: str = "exact"):
        """
        Complete distribution over prior, state and observation, p(phi, x, z). We implicitly decompose this
        hierarchically as p(phi) p(x|phi) p(z|x). Extended class to work with GMM-approximating warped meta-priors.

        Args:
            meta_prior: Meta-prior distribution, p(phi).
            observation_model: Observation model, p(z|x).
                to the transformed meta-prior.
            sample_mode: Whether to sample x from "exact" prior or approximate "prior".
                Defaults to "approximate".

        """
        super().__init__(meta_prior, observation_model)
        self.meta_prior: ApproximateWarpedGMMMetaPrior = meta_prior
        self.exact_meta_prior = meta_prior.meta_prior
        self.exact_prior = meta_prior.meta_prior.prior
        self.prior_transform = meta_prior.transform
        self.observation_model = observation_model
        self.sample_mode = sample_mode

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        phi = self.meta_prior.sample(sample_shape)
        phi_decoded = self.meta_prior.decode_sample(phi)
        prior = self.prior(**phi_decoded)
        match self.sample_mode:
            case "approximate":
                x = prior.sample()
            case "exact":
                exact_phi_decoded = self.exact_meta_prior.decode_sample(self.meta_prior.exact_prior_samples)
                x = self.exact_prior(**exact_phi_decoded).sample()
                x = self.prior_transform(x).reshape(prior.batch_shape + prior.event_shape)
            case _:
                raise ValueError('sample_mode must be one of "approximate" or "exact"')
        self.observation_model.condition_(x)
        z = self.observation_model.sample()
        return torch.cat([phi, x, z], dim=-1)

    def set_sample_mode(self, sample_mode):
        assert sample_mode in {"approximate", "exact"}, 'sample_mode must be one of "approximate" or "exact"'
        self.sample_mode = sample_mode

    def decode_sample(self, sample: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Decode tensor of sampled parameters to dictionary of tensors keyed on prior parameters phi, state x and
        observations z.

        Args:
            sample: Sampled tensor.

        Returns:
            Decoded sample.
        """
        phi = sample[..., :self.meta_prior.prior_size]
        x = sample[..., self.meta_prior.prior_size:-self.observation_model.n_observations]
        z = sample[..., -self.observation_model.n_observations:]
        return {"phi": phi,
                "x": x,
                "z": z}

    @staticmethod
    def encode_sample(decoded_sample: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode dictionary of sampled parameters to a singular tensor. Inverse operation of decode_sample.

        Args:
            decoded_sample:  Dictionary of decoded sample.

        Returns:
            Tensor encoding sample.
        """
        phi = decoded_sample["phi"]
        x = decoded_sample["x"]
        z = decoded_sample["z"]
        return torch.cat([phi, x, z], dim=-1)
