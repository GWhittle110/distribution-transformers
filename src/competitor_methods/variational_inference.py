"""
Variational inference routine
"""

import torch
from torch import Tensor
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.distributions import Distribution
from torch.distributions.utils import vec_to_tril_matrix
from torch.func import vmap, jacrev

from typing import Callable, Optional
from tqdm import tqdm

from distributions.distributions import GaussianMixtureModel, ObservationModel


class GMMVI(nn.Module):

    def __init__(self, n_components: int,
                 state_size: int,
                 prior: Distribution,
                 likelihood: dict[str, ObservationModel],
                 inverse_transform: Optional[Callable[[Tensor], Tensor]] = None):
        """
        Variational inference routine, fitting a Gaussian Mixture Model to the posterior by maximising an unbiased
        estimator of the ELBO.

        Args:
            n_components: Number of GMM components.
            state_size: Size of sample space of posterior.
            prior: Prior distribution.
            likelihood: Dictionary of likelihood distributions / observation models.
            inverse_transform: Transform from sample space of GMM approximation to prior.
                Defaults to nn.Identity().

        """
        super().__init__()
        self.state_size = state_size
        self.inverse_transform = nn.Identity() if inverse_transform is None else inverse_transform
        self.prior = prior
        self.likelihood = likelihood

        # Distribution params
        self.logits = nn.Parameter(torch.randn(*prior.batch_shape, n_components))
        self.loc = nn.Parameter(0.01 * torch.randn(*prior.batch_shape, n_components, state_size))
        self.scale_flat = nn.Parameter(0.01 * torch.randn(*prior.batch_shape, n_components,
                                                          state_size * (state_size + 1) // 2))

    def distribution(self) -> GaussianMixtureModel:
        """
        Get the current fitted distribution.

        Returns:
            Fitted GMM.

        """
        weights = self.logits.softmax(dim=-1)
        loc = self.loc
        diag = self.scale_flat[..., :self.state_size].exp()
        scale = vec_to_tril_matrix(self.scale_flat[..., self.state_size:], -1) + torch.diag_embed(diag)
        return GaussianMixtureModel(weights, loc, scale_tril=scale)

    def prior_loss(self, n_samples: int = 1,
                   distribution: Optional[Distribution] = None) -> Tensor:
        """
        KL divergence between GMM approximation for prior and the prior itself.

        Args:
            n_samples: Number of samples with which to estimate ELBO.
            distribution: External distribution to compute ELBO for. Set to None to use internal distribution.
                Defaults to None.

        Returns:
            KL divergence loss.

        """
        x = self.distribution().sample((n_samples,)) if distribution is None else distribution.sample((n_samples,))
        kl = (self.distribution().log_prob(x)
              - self.prior.log_prob(self.inverse_transform(x).reshape(n_samples, *self.prior.batch_shape,
                                                                      *self.prior.event_shape)))
        kl -= torch.logdet(vmap(jacrev(self.inverse_transform))(x.reshape(-1, self.state_size)
                                                                ).reshape(n_samples, *self.prior.batch_shape,
                                                                          self.state_size, self.state_size))
        prob = self.distribution().log_prob(x)
        kl *= torch.exp(prob - prob.clone().detach())  # Importance sampling trick
        return kl.mean(dim=0)

    def posterior_loss(self, z: dict[str, Tensor],
                       n_samples: int = 1,
                       distribution: Optional[Distribution] = None) -> Tensor:
        """
        Negative ELBO for p(x|z).

        Args:
            z: Dictionary of observation values.
            n_samples: Number of samples with which to estimate ELBO.
                Defaults to 1.
            distribution: External distribution to compute ELBO for. Set to None to use internal distribution.
                Defaults to None.

        Returns:
            Negative ELBO loss.

        """
        x = self.distribution().sample((n_samples,)) if distribution is None else distribution.sample((n_samples,))
        elbo = self.prior.log_prob(self.inverse_transform(x).reshape(n_samples, *self.prior.batch_shape,
                                                                     *self.prior.event_shape))
        for key, likelihood in self.likelihood.items():
            likelihood.condition_(self.inverse_transform(x).reshape(n_samples, *self.prior.batch_shape,
                                                                    *self.prior.event_shape))
            elbo += likelihood.log_prob(z[key]).reshape(elbo.shape)
        elbo -= self.distribution().log_prob(x)
        elbo += torch.logdet(vmap(jacrev(self.inverse_transform))(x.reshape(-1, self.state_size)
                                                                  ).reshape(n_samples, *self.prior.batch_shape,
                                                                            self.state_size, self.state_size))
        prob = self.distribution().log_prob(x)
        elbo *= torch.exp(prob - prob.clone().detach())  # Importance sampling trick
        return -elbo.mean(dim=0)

    def fit(self, z: Optional[dict[str, Tensor]] = None,
            fit_prior: bool = False,
            lr: float = 0.1,
            lr_decay: float = 0.999,
            n_iters: int = 10000,
            n_samples: int = 1,
            ewma_gamma: float = 0.01,
            progress_bar: bool = True
            ) -> dict[str, Tensor]:
        """
        Fit either prior GMM by minimising KL divergence or posterior GMM by maximising ELBO.

        Args:
            z: Dictionary of observation values. Does not need to be specified if fit_prior is True.
                Defaults to None.
            fit_prior: Whether to fit prior (True) or posterior (False).
                Defaults to posterior.
            lr: Learning rate for optimizer.
                Defaults to 0.1.
            lr_decay: Decay constant for lr per iteration.
                Defaults to 0.999.
            n_iters: Number of optimization steps to carry out.
                Defaults to 1000.
            n_samples: Number of samples with which to estimate ELBO.
                Defaults to 1.
            ewma_gamma: Decay constant for EWMA of loss.
                Defaults to 0.1.
            progress_bar: Whether to include a progress bar.
                Defaults to true.

        Returns:
            Dictionary of GMM parameters, using the scale_tril scale parametrisation.

        """
        if fit_prior:
            z = None
        assert (z is None) == fit_prior, "z must be specified if fitting posterior"

        average_loss = 0
        optimizer = Adam(self.parameters(), lr=lr)
        scheduler = ExponentialLR(optimizer, lr_decay)
        tqdm_iter = tqdm(range(n_iters), desc='VI Progress') if progress_bar else None

        for i in range(n_iters):
            tqdm_iter.update() if tqdm_iter is not None else None
            optimizer.zero_grad()
            loss = self.posterior_loss(z, n_samples) if z is not None else self.prior_loss(n_samples)
            loss.sum().backward()
            optimizer.step()
            scheduler.step()

            if tqdm_iter:
                average_loss = ((1-ewma_gamma) * average_loss + ewma_gamma * loss)
                average_loss[torch.logical_or(torch.isinf(average_loss), torch.isnan(average_loss))] = 1e6
                if fit_prior:
                    tqdm_iter.set_postfix({"Mean KL Divergence": average_loss.mean().item(),
                                           "LR": scheduler.get_last_lr()[0]})
                else:
                    tqdm_iter.set_postfix({"Mean ELBO": -average_loss.mean().item(),
                                           "LR": scheduler.get_last_lr()[0]})

        distribution = self.distribution()
        return {
            "weights": distribution.weights,
            "loc": distribution.loc,
            "scale": distribution.scale_tril
        }


if __name__ == "__main__":
    from torch.distributions import InverseGamma
    from distributions.distributions import ScaleGaussianObservationModel
    from distributions.utils import plot_distributions

    from time import time


    n = torch.Size()
    device = torch.device("cpu:0")
    prior = InverseGamma(torch.ones(n, device=device), torch.ones(n, device=device))
    likelihood = {"obs_1": ScaleGaussianObservationModel(loc=torch.ones(*n, 1, device=device))}
    vi = GMMVI(5, 1, prior, likelihood, inverse_transform=torch.exp).to(device)
    z = {"obs_1": torch.ones(*n, 1, device=device)}
    print(vi.prior_loss(100000).mean())
    t0 = time()
    vi.fit(z, fit_prior=True, n_samples=1000, lr=0.1, n_iters=10000)
    print(time()-t0)
    print(vi.prior_loss(100000).mean())
    print(vi.distribution().weights)
    print(vi.distribution().loc)
    print(vi.distribution().scale_tril)
    with torch.no_grad():
        prior = InverseGamma(torch.ones(n), torch.ones(n))
        plot_distributions(prior, vi.cpu().distribution(), None, torch.log, bounds=(1e-6, 5))
