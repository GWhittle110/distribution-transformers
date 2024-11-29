"""
Utility functions for distributions
"""

import torch
from torch import Tensor
from torch.distributions import Distribution, InverseGamma, Normal
from torch.func import vmap, jacrev

from math import sqrt
from typing import Optional, Callable
import matplotlib.pyplot as plt


def decode_gmm_sample(sample: Tensor, scale_parametrisation: str = "covariance_matrix"):
    """
    Decode a sequence representation GMM sample into a dict of parameters.

    Args:
        sample: Sequence representation GMM.
        scale_parametrisation: Parametrisation used for scale parameter. Must be one of "covariance_matrix",
            "precision_matrix" or "scale_tril".
            Defaults to "covariance_matrix".

    Returns:
        Dict representation GMM.

    """
    state_size = int(sqrt(sample.shape[-1]))
    weights = sample[..., 0]
    loc = sample[..., 1:state_size+1]
    scale = sample[..., -state_size ** 2:].reshape(*sample.shape[:-1], state_size, state_size)
    return {"weights": weights, "loc": loc, scale_parametrisation: scale}


def encode_gmm_sample(sample: dict[str, Tensor], scale_parametrisation: str = "covariance_matrix"):
    """
    Encode a parameter dict representation GMM sample as a sequence representation.

    Args:
        sample: Dict representation GMM.
        scale_parametrisation: Parametrisation used for scale parameter. Must be one of "covariance_matrix",
            "precision_matrix" or "scale_tril".
            Defaults to "covariance_matrix".

    Returns:
        Sequence representation GMM.

    """
    return torch.cat([sample["weights"].unsqueeze(-1), sample["loc"], sample[scale_parametrisation].flatten(-2)],
                     dim=-1)


def kl_divergence(p: Distribution, q: Distribution,
                  q_transform: Optional[Callable[[Tensor], Tensor]] = None,
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
    q_samples = q_samples.reshape((n_samples,) + q.batch_shape + q.event_shape)
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


def plot_distributions(p: Distribution, q: Optional[Distribution] = None,
                       q_transform: Optional[Callable[[Tensor], Tensor]] = None,
                       bounds: tuple[float, float] = (-5., 5.),
                       n_points: int = 1000) -> plt.Figure:
    """
    Function to plot a (1 dimensional) distribution, or a pair of (1 dimensional) distributions.

    Example:
        >>> p = InverseGamma(1, 1)
        >>> q = Normal(0, 1)
        >>> plot_distributions(p, q, torch.log, (0.01, 5))

    Args:
        p: First distribution to plot.
        q: Second distribution to plot.
        q_transform: Transform mapping sample space of p to sample space of q.
        bounds: Tuple of upper and lower bounds.
        n_points: Number of points at which to evaluate density. Uniformly distributed in bounds.

    Returns:
        Figure object.

    """
    points = torch.linspace(*bounds, steps=n_points)
    p_density = p.log_prob(points.reshape((n_points,) + p.event_shape)).exp()
    fig, ax = plt.subplots()
    ax.plot(points, p_density)

    if q is not None:
        if q_transform is None:
            q_transform = torch.nn.Identity()
        q_density = torch.exp(q.log_prob(q_transform(points).reshape((n_points,) + q.event_shape))
                              + torch.log(vmap(jacrev(q_transform))(points).reshape((-1,) + p.batch_shape)))
        ax.plot(points, q_density)
        ax.legend(["p", "q"])
        ax.annotate(f"KL Divergence: {kl_divergence(p, q, q_transform):5.4f}", (0.6, 0.9), xycoords="axes fraction")

    ax.set_title("Density plot")
    ax.set_ylabel("Density")
    plt.show()
    return fig
