"""
Custom distributions used by the model for meta-priors, priors, likelihoods and sampling
"""

import torch
from torch.distributions import (Distribution, Wishart, MultivariateNormal, Independent, Categorical,
                                 MixtureSameFamily, Dirichlet, constraints)
from torch.distributions.utils import lazy_property
from torch.types import _size
from typing import Optional, Callable


class GaussianMixtureModel(MixtureSameFamily):
    arg_constraints = {
        "weights": constraints.simplex,
        "loc": constraints.real_vector,
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }
    support = constraints.real_vector

    def __init__(self, weights: torch.Tensor, loc: torch.Tensor, covariance_matrix: Optional[torch.Tensor] = None,
                 precision_matrix: Optional[torch.Tensor] = None, scale_tril: Optional[torch.Tensor] = None):
        """
        Gaussian Mixture Model distribution. Supports sampling with different parameters in dimension 0.

        Example - no batching:
            >>> weights = torch.ones(3) / 3
            >>> loc = torch.zeros(3, 2, dtype=torch.float32)
            >>> covariance_matrix = torch.eye(2, dtype=torch.float32).broadcast_to(3, 2, 2)
            >>> gmm = GaussianMixtureModel(weights, loc, covariance_matrix)
            >>> print(gmm.sample())

        Example - batching:
            >>> weights = torch.ones(4, 3) / 3
            >>> loc = torch.zeros(4, 3, 2, dtype=torch.float32)
            >>> covariance_matrix = torch.eye(2, dtype=torch.float32).broadcast_to(4, 3, 2, 2)
            >>> gmm = GaussianMixtureModel(weights, loc, covariance_matrix)
            >>> print(gmm.sample())

        Args:
            weights: Weighting of each mixture component. Must all be positive and sum to 1.
            loc: torch.Tensor of means of each mixture component.
            covariance_matrix: Tensor of covariance matrices of each mixture component. Must all be positive definite.
            precision_matrix:  Tensor of precision matrices of each mixture component. Must all be positive definite.
            scale_tril: torch.Tensor of lower triangular representation of scale matrix, i.e. Cholesky decomposition of
                covariance matrix. Must have positive diagonal elements,
        """
        super().__init__(Categorical(weights),
                         Independent(MultivariateNormal(loc,
                                                        covariance_matrix=covariance_matrix,
                                                        precision_matrix=precision_matrix,
                                                        scale_tril=scale_tril),
                                     0))
        self.weights = weights
        self.loc = loc
        self.covariance_matrix = covariance_matrix
        self.precision_matrix = precision_matrix
        self.scale_tril = scale_tril
        self.n_components = weights.shape[-1]
        self.state_size = loc.shape[-1]
        if (covariance_matrix is not None) + (scale_tril is not None) + (
                precision_matrix is not None
        ) != 1:
            raise ValueError(
                "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."
            )

        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError(
                    "scale_tril matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            self.scale_parametrisation = "scale_tril"
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError(
                    "covariance_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            self.scale_parametrisation = "covariance_matrix"
        elif precision_matrix is not None:
            if precision_matrix.dim() < 2:
                raise ValueError(
                    "precision_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            self.scale_parametrisation = "precision_matrix"

    @lazy_property
    def weights(self):
        return self.weights

    @lazy_property
    def loc(self):
        return self.loc

    @lazy_property
    def covariance_matrix(self):
        return self.covariance_matrix

    @lazy_property
    def precision_matrix(self):
        return self.precision_matrix

    @lazy_property
    def scale_tril(self):
        return self.scale_tril


class MetaPrior(Distribution):
    """
    Abstract base class of meta-priors p(phi) where phi parametrises a prior over x, that is to say priors over priors.
    """

    def __init__(self, prior: type[Distribution]):
        super().__init__()
        self.prior = prior
        self.prior_size: Optional[int] = None

    def decode_sample(self, sample: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Decode tensor of sampled parameters to dictionary of tensors keyed by parameter.

        Args:
            sample: Sampled tensor

        Returns:
            Decoded sample
        """
        raise NotImplementedError

    def encode_sample(self, decoded_sample: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode dictionary of sampled parameters to a singular tensor. Inverse operation of decode_sample

        Args:
            decoded_sample:  Dictionary of decoded sample

        Returns:
            Tensor encoding sample
        """
        raise NotImplementedError


class GaussianMixtureModelConjugateMetaPrior(MetaPrior):
    arg_constraints = {
        "weights_concentration": constraints.simplex,
        "loc_loc": constraints.real_vector,
        "loc_covariance_matrix": constraints.positive_definite,
        "loc_precision_matrix": constraints.positive_definite,
        "loc_scale_tril": constraints.lower_cholesky,
        "scale_df": constraints.greater_than(0),
        "scale_loc": constraints.real_vector,
        "scale_covariance_matrix": constraints.positive_definite,
        "scale_precision_matrix": constraints.positive_definite,
        "scale_scale_tril": constraints.lower_cholesky,
        "scale_eps": constraints.greater_than_eq(0)
    }
    has_rsample = True

    def __init__(self,
                 weights_concentration: Optional[torch.Tensor] = None,
                 loc_loc: Optional[torch.Tensor] = None,
                 loc_covariance_matrix: Optional[torch.Tensor] = None,
                 loc_precision_matrix: Optional[torch.Tensor] = None,
                 loc_scale_tril: Optional[torch.Tensor] = None,
                 scale_parametrisation: Optional[str] = None,
                 scale_df: Optional[torch.Tensor] = None,
                 scale_covariance_matrix: Optional[torch.Tensor] = None,
                 scale_precision_matrix: Optional[torch.Tensor] = None,
                 scale_scale_tril: Optional[torch.Tensor] = None,
                 scale_eps: Optional[float] = None,
                 n_components: Optional[int] = None,
                 state_size: Optional[int] = None,
                 ):
        """
        Meta-prior for a Gaussian Mixture Model. Uses conjugate priors for all parameters. Each parameter is referred to
        as for a single distribution, but must be provided as a tensor of parameters for each component.

        Example - default values:

            >>> gmm_conjugate_meta_prior = GaussianMixtureModelConjugateMetaPrior(n_components=3, state_size=2)
            >>> sample = gmm_conjugate_meta_prior.sample((1,))
            >>> print(gmm_conjugate_meta_prior.decode_sample(sample))

        Args:
            weights_concentration: Concentration parameter for Dirichlet prior over weights.
                Defaults to uniform.
            loc_loc: Loc parameter for Multivariate Normal prior over loc.
                Defaults to zero for all components.
            loc_covariance_matrix: Covariance matrix parameter for Multivariate Normal prior over loc.
                Defaults to identity for all components.
            loc_precision_matrix: Precision matrix parameter for Multivariate Normal prior over loc.
                Defaults to identity for all components.
            loc_scale_tril: Lower triangular scale parameter for Multivariate Normal prior over loc.
                Defaults to identity for all components.
            scale_parametrisation: Parametrisation used for scale parameter, common between all components.
                Defaults to "covariance_matrix".
            scale_df: Degrees of freedom of Wishart prior for precision_matrix.
                Defaults to state_size + 1 for all components.
            scale_covariance_matrix: Covariance matrix of Wishart prior for precision_matrix.
                Defaults to identity for all components.
            scale_precision_matrix: Precision matrix of Wishart prior for precision_matrix.
                Defaults to identity for all components.
            scale_scale_tril: Lower triangular scale matrix of Wishart prior for precision_matrix.
                Defaults to identity for all components.
            scale_eps: Small constant to add to diagonal of sampled scale matrix.
                Defaults to 1e-6.
            n_components: Number of components in mixture model. Only specify if using default value for
                weights_concentration else overridden.
            state_size: Size of state in mixture model. Only specify if using default value for loc_loc else overridden.
        """
        super().__init__(GaussianMixtureModel)

        if n_components is not None or state_size is not None:
            assert n_components is not None and state_size is not None, \
                "both n_components and state_size must be specified if using default values"
        self.n_components = weights_concentration.shape[-1] if weights_concentration is not None else n_components
        self.state_size = loc_loc.shape[-1] if loc_loc is not None else state_size
        self.weights_concentration = torch.ones(self.n_components) / self.n_components \
            if weights_concentration is None else weights_concentration
        self.loc_loc = torch.zeros((self.n_components, self.state_size), dtype=torch.float32) \
            if loc_loc is None else loc_loc
        if loc_covariance_matrix is None and loc_precision_matrix is None and loc_scale_tril is None:
            self.loc_covariance_matrix = torch.eye(self.state_size, dtype=torch.float32
                                                   ).broadcast_to(self.n_components, self.state_size, self.state_size)
            self.loc_precision_matrix = loc_precision_matrix
            self.loc_scale_tril = loc_scale_tril
        else:
            self.loc_covariance_matrix = loc_covariance_matrix
            self.loc_precision_matrix = loc_precision_matrix
            self.loc_scale_tril = loc_scale_tril
        self.scale_parametrisation = "covariance_matrix" if scale_parametrisation is None else scale_parametrisation
        self.scale_df = self.state_size + 1 if scale_df is None else scale_df
        if scale_covariance_matrix is None and scale_precision_matrix is None and scale_scale_tril is None:
            self.scale_covariance_matrix = torch.eye(self.state_size, dtype=torch.float32
                                                     ).broadcast_to(self.n_components, self.state_size, self.state_size)
            self.scale_precision_matrix = scale_precision_matrix
            self.scale_scale_tril = scale_scale_tril
        else:
            self.scale_covariance_matrix = scale_covariance_matrix
            self.scale_precision_matrix = scale_precision_matrix
            self.scale_scale_tril = scale_scale_tril
        self.scale_eps = 1e-6 if scale_eps is None else scale_eps
        self.prior_size = self.n_components * (1 + self.state_size + self.state_size ** 2)

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        weights = Dirichlet(self.weights_concentration).sample(sample_shape)
        loc = Independent(MultivariateNormal(self.loc_loc,
                                             covariance_matrix=self.loc_covariance_matrix,
                                             precision_matrix=self.loc_precision_matrix,
                                             scale_tril=self.loc_scale_tril
                                             ), 0).sample(sample_shape)
        precision_matrix = Independent(Wishart(self.scale_df,
                                               covariance_matrix=self.scale_covariance_matrix,
                                               precision_matrix=self.scale_precision_matrix,
                                               scale_tril=self.scale_scale_tril
                                               ), 0).sample(sample_shape)
        # Add jitter of magnitude scale_eps to diagonal
        precision_matrix += self.scale_eps * torch.eye(self.state_size)
        match self.scale_parametrisation:
            case "covariance_matrix":
                scale = torch.linalg.inv(precision_matrix.to(torch.float64)).to(torch.float32)
            case "precision_matrix":
                scale = precision_matrix
            case "scale_tril":
                scale = torch.linalg.cholesky(torch.linalg.inv(precision_matrix))
            case _:
                raise AssertionError('scale_parametrisation must be one of "covariance_matrix", "precision_matrix" or '
                                     '"scale_tril"')
        return torch.cat([weights, loc.flatten(start_dim=-2), scale.flatten(start_dim=-3)], dim=-1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Calculate the logarithm of the probability density function evaluated at the input value.
        Args:
            value: Value to query probability density function.

        Returns:
            Log probability of value.

        """
        params_dict = self.decode_sample(value)

        match self.scale_parametrisation:
            case "covariance_matrix":
                precision_matrix = torch.linalg.inv(params_dict[self.scale_parametrisation])
            case "precision_matrix":
                precision_matrix = params_dict[self.scale_parametrisation]
            case "scale_tril":
                covariance_matrix = torch.einsum("...ij, ...kj -> ...ik", params_dict[self.scale_parametrisation],
                                                 params_dict[self.scale_parametrisation])
                precision_matrix = torch.linalg.inv(covariance_matrix)
            case _:
                raise AssertionError('scale_parametrisation must be one of "covariance_matrix", "precision_matrix" or '
                                     '"scale_tril"')
        loc = params_dict["loc"].reshape(value.shape[:-1] + (self.n_components, self.state_size))
        precision_matrix = precision_matrix.reshape(value.shape[:-1] + (self.n_components, self.state_size,
                                                                        self.state_size))
        return (Dirichlet(self.weights_concentration).log_prob(params_dict["weights"]) +
                Independent(MultivariateNormal(self.loc_loc,
                                               covariance_matrix=self.loc_covariance_matrix,
                                               precision_matrix=self.loc_precision_matrix,
                                               scale_tril=self.loc_scale_tril
                                               ), 1).log_prob(loc) +
                Independent(Wishart(self.scale_df,
                                    covariance_matrix=self.scale_covariance_matrix,
                                    precision_matrix=self.scale_precision_matrix,
                                    scale_tril=self.scale_scale_tril
                                    ), 1).log_prob(precision_matrix))

    def decode_sample(self, sample: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Decode tensor of sampled parameters to dictionary of tensors keyed by GMM parameter.

        Args:
            sample: Sampled tensor

        Returns:
            Decoded sample
        """
        sample_shape = sample.shape[:-1]
        weights = sample[..., :self.n_components]
        loc = sample[..., self.n_components:self.n_components*(1+self.state_size)].reshape(*sample_shape,
                                                                                           self.n_components,
                                                                                           self.state_size)
        scale = sample[..., self.n_components*(1+self.state_size):].reshape(*sample_shape,
                                                                            self.n_components,
                                                                            self.state_size,
                                                                            self.state_size)
        return {"weights": weights,
                "loc": loc,
                self.scale_parametrisation: scale}

    def encode_sample(self, decoded_sample: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode dictionary of sampled parameters to a singular tensor. Inverse operation of decode_sample

        Args:
            decoded_sample:  Dictionary of decoded sample

        Returns:
            Tensor encoding sample
        """
        weights = decoded_sample["weights"]
        loc = decoded_sample["loc"]
        scale = decoded_sample[self.scale_parametrisation]
        return torch.cat([weights, loc.flatten(start_dim=-2), scale.flatten(start_dim=-3)], dim=-1)

    @lazy_property
    def weights_concentration(self):
        return self.weights_concentration

    @lazy_property
    def loc_loc(self):
        return self.loc_loc

    @lazy_property
    def loc_covariance_matrix(self):
        return self.loc_covariance_matrix

    @lazy_property
    def loc_precision_matrix(self):
        return self.loc_precision_matrix

    @lazy_property
    def loc_scale_tril(self):
        return self.loc_scale_tril

    @lazy_property
    def scale_df(self):
        return self.scale_df

    @lazy_property
    def scale_loc(self):
        return self.scale_loc

    @lazy_property
    def scale_covariance_matrix(self):
        return self.scale_covariance_matrix

    @lazy_property
    def scale_precision_matrix(self):
        return self.scale_precision_matrix

    @lazy_property
    def scale_scale_tril(self):
        return self.scale_scale_tril

    @lazy_property
    def scale_eps(self):
        return self.scale_eps


class ObservationModel(Distribution):
    has_rsample = True

    def __init__(self):
        """
        Abstract base class of observation models p(z|x).
        """
        super().__init__()
        self.distribution: Optional[Distribution] = None
        self.n_observations: Optional[int] = None

    def condition_(self, x: torch.Tensor):
        """
        Condition observation distribution on state
        Args:
            x: State to condition sample on. Can be batched or not.

        """

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        return self.distribution.sample(sample_shape=sample_shape)


class DirectGaussianObservationModel(ObservationModel):
    arg_constraints = {
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }

    def __init__(self, covariance_matrix: Optional[torch.Tensor] = None,
                 precision_matrix: Optional[torch.Tensor] = None, scale_tril: Optional[torch.Tensor] = None):
        """
        Observation model for direction observation of state subject to Gaussian noise

        Example - no batching:
            >>> x = torch.ones(2, dtype=torch.float32)
            >>> covariance_matrix = torch.eye(2, dtype=torch.float32)
            >>> dgom = DirectGaussianObservationModel(covariance_matrix=covariance_matrix)
            >>> dgom.condition_(x)
            >>> print(dgom.sample())

        Example - batching:
            >>> x = torch.ones((2, 2), dtype=torch.float32)
            >>> covariance_matrix = torch.eye(2, dtype=torch.float32).broadcast_to(2, 2, 2)
            >>> dgom = DirectGaussianObservationModel(covariance_matrix=covariance_matrix)
            >>> dgom.condition_(x)
            >>> print(dgom.sample())

        Args:
            covariance_matrix: Covariance matrix for Gaussian noise.
            precision_matrix: Precision matrix for Gaussian noise.
            scale_tril: Lower triangular scale parameter for Gaussian noise.

        """
        super().__init__()
        if (covariance_matrix is not None) + (scale_tril is not None) + (
                precision_matrix is not None
        ) != 1:
            raise ValueError(
                "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."
            )

        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError(
                    "scale_tril matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            self.scale_parametrisation = "scale_tril"
            self.n_observations = scale_tril.shape[-1]
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError(
                    "covariance_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            self.scale_parametrisation = "covariance_matrix"
            self.n_observations = covariance_matrix.shape[-1]
        elif precision_matrix is not None:
            if precision_matrix.dim() < 2:
                raise ValueError(
                    "precision_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            self.scale_parametrisation = "precision_matrix"
            self.n_observations = precision_matrix.shape[-1]

        self.distribution: Optional[MultivariateNormal] = None
        self.covariance_matrix = covariance_matrix
        self.precision_matrix = precision_matrix
        self.scale_tril = scale_tril

    def condition_(self, x: torch.Tensor):
        """
        Condition observation distribution on state
        Args:
            x: State to condition sample on. Can be batched or not.

        """
        self.distribution = MultivariateNormal(loc=x, covariance_matrix=self.covariance_matrix,
                                               precision_matrix=self.precision_matrix, scale_tril=self.scale_tril)

    @lazy_property
    def covariance_matrix(self):
        return self.covariance_matrix

    @lazy_property
    def precision_matrix(self):
        return self.precision_matrix

    @lazy_property
    def scale_tril(self):
        return self.scale_tril


class MappedGaussianObservationModel(DirectGaussianObservationModel):
    arg_constraints = {
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }

    def __init__(self, covariance_matrix: Optional[torch.Tensor] = None,
                 precision_matrix: Optional[torch.Tensor] = None, scale_tril: Optional[torch.Tensor] = None,
                 mapping: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        """
        Observation model for observation of mapping of state subject to Gaussian noise

        Example - no batching:
            >>> x = torch.ones(2, dtype=torch.float32)
            >>> covariance_matrix = torch.eye(2, dtype=torch.float32)
            >>> mapping = lambda x: x ** 2
            >>> mgom = MappedGaussianObservationModel(covariance_matrix=covariance_matrix, mapping=mapping)
            >>> mgom.condition_(x)
            >>> print(mgom.sample())

        Example - batching:
            >>> x = torch.ones((2, 2), dtype=torch.float32)
            >>> covariance_matrix = torch.eye(2, dtype=torch.float32).broadcast_to(2, 2, 2)
            >>> mapping = lambda x: x ** 3
            >>> mgom = MappedGaussianObservationModel(covariance_matrix=covariance_matrix, mapping=mapping)
            >>> mgom.condition_(x)
            >>> print(mgom.sample())

        Args:
            covariance_matrix: Covariance matrix for Gaussian noise.
            precision_matrix: Precision matrix for Gaussian noise.
            scale_tril: Lower triangular scale parameter for Gaussian noise.
            mapping: Mapping from state to observation.
                Defaults to identity.

        """
        super().__init__(covariance_matrix, precision_matrix, scale_tril)
        self.mapping = torch.nn.Identity() if mapping is None else mapping

    def condition_(self, x: torch.Tensor):
        super().condition_(self.mapping(x))


class LinearGaussianObservationModel(MappedGaussianObservationModel):
    arg_constraints = {
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }

    def __init__(self, observation_matrix: torch.Tensor = None, covariance_matrix: Optional[torch.Tensor] = None,
                 precision_matrix: Optional[torch.Tensor] = None, scale_tril: Optional[torch.Tensor] = None):
        """
        Observation model for observation of mapping of state subject to Gaussian noise

        Example - no batching:
            >>> x = torch.ones(2, dtype=torch.float32)
            >>> covariance_matrix = torch.eye(3, dtype=torch.float32)
            >>> observation_matrix = torch.ones((3, 2), dtype=torch.float32)
            >>> mgom = LinearGaussianObservationModel(observation_matrix, covariance_matrix=covariance_matrix)
            >>> mgom.condition_(x)
            >>> print(mgom.sample())

        Example - batching:
            >>> x = torch.ones((2, 2), dtype=torch.float32)
            >>> covariance_matrix = torch.eye(3, dtype=torch.float32).broadcast_to(2, 3, 3)
            >>> observation_matrix = torch.ones((3, 2), dtype=torch.float32)
            >>> mgom = LinearGaussianObservationModel(observation_matrix, covariance_matrix=covariance_matrix)
            >>> mgom.condition_(x)
            >>> print(mgom.sample())

        Args:
            observation_matrix: Matrix mapping from state to observation
            covariance_matrix: Covariance matrix for Gaussian noise.
            precision_matrix: Precision matrix for Gaussian noise.
            scale_tril: Lower triangular scale parameter for Gaussian noise.

        """
        if covariance_matrix is not None:
            n_obs = covariance_matrix.shape[-1]
        elif precision_matrix is not None:
            n_obs = precision_matrix.shape[-1]
        elif scale_tril is not None:
            n_obs = scale_tril.shape[-1]
        else:
            raise ValueError(
                "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."
            )

        assert observation_matrix.dim() == 2, \
            "Observation matrix must be exactly 2 dimensional, batching is not supported"
        assert observation_matrix.shape[-2] == n_obs
        self.observation_matrix = observation_matrix
        mapping = lambda x: torch.einsum("ij,...j->...i", self.observation_matrix, x)
        super().__init__(covariance_matrix=covariance_matrix,
                         precision_matrix=precision_matrix,
                         scale_tril=scale_tril,
                         mapping=mapping)


class CompleteDistribution(Distribution):
    has_rsample = True
    _validate_args = False

    def __init__(self, meta_prior: MetaPrior, observation_model: ObservationModel):
        """
        Complete distribution over prior, state and observation, p(phi, x, z). We implicitly decompose this
        hierarchically as p(phi) p(x|phi) p(z|x).

        Example - no batching:
            >>> meta_prior = GaussianMixtureModelConjugateMetaPrior(n_components=4, state_size=2)
            >>> covariance_matrix = torch.eye(2, dtype=torch.float32)
            >>> observation_model = DirectGaussianObservationModel(covariance_matrix=covariance_matrix)
            >>> complete_distribution = CompleteDistribution(meta_prior, observation_model)
            >>> print(complete_distribution.sample())

        Example - batching:
            >>> meta_prior = GaussianMixtureModelConjugateMetaPrior(n_components=4, state_size=2)
            >>> covariance_matrix = torch.eye(2, dtype=torch.float32)
            >>> observation_model = DirectGaussianObservationModel(covariance_matrix=covariance_matrix)
            >>> complete_distribution = CompleteDistribution(meta_prior, observation_model)
            >>> print(complete_distribution.sample((5, 10)))

        Args:
            meta_prior: Meta-prior distribution, p(phi)
            observation_model: Observation model, p(z|x)
        """
        super().__init__()
        self.meta_prior = meta_prior
        self.prior = meta_prior.prior
        self.observation_model = observation_model

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        phi = self.meta_prior.sample(sample_shape)
        phi_decoded = self.meta_prior.decode_sample(phi)
        x = self.prior(**phi_decoded).sample()
        self.observation_model.condition_(x)
        z = self.observation_model.sample()
        return torch.cat([phi, x, z], dim=-1)

    def decode_sample(self, sample: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Decode tensor of sampled parameters to dictionary of tensors keyed on prior parameters phi, state x and
        observations z.

        Args:
            sample: Sampled tensor

        Returns:
            Decoded sample
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
        Encode dictionary of sampled parameters to a singular tensor. Inverse operation of decode_sample

        Args:
            decoded_sample:  Dictionary of decoded sample

        Returns:
            Tensor encoding sample
        """
        phi = decoded_sample["phi"]
        x = decoded_sample["x"]
        z = decoded_sample["z"]
        return torch.cat([phi, x, z], dim=-1)
