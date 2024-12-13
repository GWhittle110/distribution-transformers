"""
Testing workflow
"""

import torch
from torch import Tensor

from time import time
from typing import Callable, Optional

from model.distribution_transformer import DistributionTransformer
from distributions.distributions import CompleteDistribution, GaussianMixtureModel
from distributions.utils import decode_gmm_sample, kl_divergence, plot_distributions
from competitor_methods.variational_inference import GMMVI


def test_conjugate_prior(model: DistributionTransformer,
                         complete_distribution: CompleteDistribution,
                         conjugacy_update: Callable[[dict[str, Tensor], dict[str, Tensor], str], dict[str, Tensor]],
                         n_test_priors: int = 1000,
                         n_kl_samples: int = 10000,
                         plot: bool = False,
                         bounds_func: Optional[Callable[[dict[str, Tensor]], tuple[float, float]]] = None,
                         gpu_device: str = "cuda:0",
                         _run=None
                         ) -> None:
    """
    Standard testing routine for experiments involving conjugate priors

    Args:
        model: Model to Test.
        complete_distribution: Complete distribution over priors, state and observation.
        conjugacy_update: Function taking a dict of prior parameters, a dict of observations, and returning a dict of
            posterior parameters.
        n_test_priors: Number of priors to test model with.
            Defaults to 1000.
        n_kl_samples: Number of samples to take when computing KL divergences.
            Defaults to 10000.
        plot: Whether to plot.
            Defaults to False.
        gpu_device: GPU device.
            Defaults to "cuda:0".
        bounds_func: Function to calculate plotting bounds from exact distribution parameters.
            Defaults to an estimate of the 5-95%ile from 10000 samples
        _run: Sacred run object.

    """

    with torch.no_grad():
        device = gpu_device if torch.cuda.is_available() else 'cpu:0'

        # Test inputs
        phi, x, z = complete_distribution.sample((n_test_priors,))

        # Device
        phi = phi.to(device)
        z = {key: val.to(device) for key, val in z.items()}
        model = model.to(device)

        # Exact solution
        phi_prior_dict = complete_distribution.meta_prior.decode_sample(phi)
        exact_prior = complete_distribution.meta_prior.prior(**phi_prior_dict)
        phi_posterior_dict = conjugacy_update(phi_prior_dict, z, device)
        exact_posterior = complete_distribution.meta_prior.prior(**phi_posterior_dict)

        # Model solution
        start_time = time()
        phi_in, phi_out = model(phi.to(device), **z)
        inference_time = time() - start_time
        model_prior = GaussianMixtureModel(**decode_gmm_sample(phi_in))
        model_posterior = GaussianMixtureModel(**decode_gmm_sample(phi_out))

        prior_kl_divergence = kl_divergence(exact_prior, model_prior, model.sample_space_transform,
                                            n_kl_samples).mean().item()
        posterior_kl_divergence = kl_divergence(exact_posterior, model_posterior, model.sample_space_transform,
                                                n_kl_samples).mean().item()

        print(f"GMM approximation prior mean KL divergence: {prior_kl_divergence.mean().item()}\n"
              f"Posterior mean KL divergence: {posterior_kl_divergence.mean().item()}\n")

        # Single problem run

        # Test inputs
        phi, x, z = complete_distribution.sample()

        # Device
        model.cpu()

        # Exact solution
        phi_prior_dict = complete_distribution.meta_prior.decode_sample(phi)
        exact_prior = complete_distribution.meta_prior.prior(**phi_prior_dict)
        phi_posterior_dict = conjugacy_update(phi_prior_dict, z, "cpu")
        exact_posterior = complete_distribution.meta_prior.prior(**phi_posterior_dict)

        # Model solution
        start_time = time()
        phi_in, phi_out = model(phi, **z)
        single_inference_time = time() - start_time
        model_prior = GaussianMixtureModel(**decode_gmm_sample(phi_in))
        model_posterior = GaussianMixtureModel(**decode_gmm_sample(phi_out))

        if _run is not None:
            _run.info.update({
                "prior_kl_divergence": prior_kl_divergence,
                "posterior_kl_divergence": posterior_kl_divergence,
                "inference_time": inference_time,
                "single_inference_time": single_inference_time
            })

        # Plotting
        if plot:
            assert torch.prod(torch.tensor(exact_prior.event_shape)).item() == 1, \
                "plotting only supported for single output distributions"

            if bounds_func is None:
                def bounds_func(params_dict: dict[str, Tensor]) -> tuple[float, float]:
                    dist = complete_distribution.meta_prior.prior(**params_dict)
                    samples = dist.sample((10000,))
                    samples = samples.sort().values
                    return samples[499].item(), samples[9499].item()


            prior_plot = plot_distributions(exact_prior, model_prior, None, model.sample_space_transform,
                                            bounds_func(phi_prior_dict),
                                            n_kl_samples=n_kl_samples)

            posterior_plot = plot_distributions(exact_posterior, model_posterior, None, model.sample_space_transform,
                                                bounds_func(phi_posterior_dict),
                                                n_kl_samples=n_kl_samples)

            if _run is not None:
                prior_plot.savefig(_run.observers[0].dir + "\\prior_plot.png")
                posterior_plot.savefig(_run.observers[0].dir + "\\posterior_plot.png")


def test(model: DistributionTransformer,
         complete_distribution: CompleteDistribution,
         competitor_kwargs: Optional[dict[str, dict]] = None,
         inverse_transform: Optional[Callable[[Tensor], Tensor]] = None,
         n_test_priors: int = 1000,
         n_kl_samples: int = 10000,
         plot: bool = False,
         bounds_func: Optional[Callable[[dict[str, Tensor]], tuple[float, float]]] = None,
         gpu_device: str = "cuda:0",
         _run=None
         ) -> None:
    """
    Standard testing routine for experiments not involving conjugate priors

    Args:
        model: Model to Test.
        complete_distribution: Complete distribution over priors, state and observation.
        competitor_kwargs: Dictionary of dictionaries of parameters for competitor methods.
            Defaults to None.
        inverse_transform: Transform from sample space of GMM approximation to prior.
            Defaults to None.
        n_test_priors: Number of priors to test model with.
            Defaults to 1000.
        n_kl_samples: Number of samples to take when computing KL divergences.
            Defaults to 10000.
        plot: Whether to plot.
            Defaults to False.
        gpu_device: GPU device.
            Defaults to "cuda:0".
        bounds_func: Function to calculate plotting bounds from exact distribution parameters.
            Defaults to an estimate of the 5-95%ile from 10000 samples
        _run: Sacred run object.

    """

    with torch.no_grad():
        competitor_kwargs = dict() if competitor_kwargs is None else competitor_kwargs

        device = gpu_device if torch.cuda.is_available() else 'cpu:0'

        # Test inputs
        phi, x, z = complete_distribution.sample((n_test_priors,))

        # Device
        phi = phi.to(device)
        z = {key: val.to(device) for key, val in z.items()}
        model = model.to(device)

        # Exact prior
        phi_prior_dict = complete_distribution.meta_prior.decode_sample(phi)
        prior = complete_distribution.meta_prior.prior(**phi_prior_dict)

        # Model solution
        start_time = time()
        phi_in, phi_out = model(phi.to(device), **z)
        inference_time = time() - start_time
        model_prior = GaussianMixtureModel(**decode_gmm_sample(phi_in))
        model_posterior = GaussianMixtureModel(**decode_gmm_sample(phi_out))

        prior_kl_divergence = kl_divergence(prior, model_prior, model.sample_space_transform,
                                            n_kl_samples)
        model_expected_prior_kl_divergence = prior_kl_divergence.mean().item()
        model_std_prior_kl_divergence = prior_kl_divergence.std().item()

        model_posterior_nll = -model_posterior.log_prob(x.reshape(model_posterior.batch_shape
                                                                  + model_posterior.event_shape).to(device)
                                                        )
        model_posterior_expected_nll = model_posterior_nll.mean().item()
        model_posterior_std_nll = model_posterior_nll.std().item()

        print(f"GMM approximation prior mean KL divergence: {prior_kl_divergence}")

        if "vi" in competitor_kwargs:
            # VI solution
            before_vi = time()
            vi = GMMVI(model.n_components, model.state_size, prior, complete_distribution.observation_model,
                       inverse_transform).to(device)
            torch.set_grad_enabled(True)
            vi.fit(z, **competitor_kwargs["vi"])
            torch.set_grad_enabled(False)
            vi_time = time() - before_vi
            vi_posterior = vi.distribution()
            vi_posterior_kl_divergence = kl_divergence(vi_posterior, model_posterior,
                                                       n_samples=n_kl_samples).mean().item()
            print(f"VI - model mean KL divergence: {vi_posterior_kl_divergence}")
            vi_nll = -vi.distribution().log_prob(x.reshape(model_posterior.batch_shape
                                                           + model_posterior.event_shape).to(device))
            vi_expected_nll = vi_nll.mean().item()
            vi_std_nll = vi_nll.mean().item()
            vi_elbo = -vi.posterior_loss(z, n_samples=n_test_priors)
            vi_expected_elbo = vi_elbo.mean().item()
            vi_std_elbo = vi_elbo.std().item()
            model_elbo = -vi.posterior_loss(z, n_samples=n_test_priors, distribution=model_posterior)
            model_expected_elbo = model_elbo.mean().item()
            model_std_elbo = model_elbo.std().item()

        # Single problem run

        # Test inputs
        phi, x, z = complete_distribution.sample()

        # Device
        model.cpu()

        phi_prior_dict = complete_distribution.meta_prior.decode_sample(phi)
        prior = complete_distribution.meta_prior.prior(**phi_prior_dict)

        # Model solution
        start_time = time()
        phi_in, phi_out = model(phi, **z)
        single_inference_time = time() - start_time
        model_prior = GaussianMixtureModel(**decode_gmm_sample(phi_in))
        model_posterior = GaussianMixtureModel(**decode_gmm_sample(phi_out))

        if _run is not None:
            _run.info.update({
                "model_expected_prior_kl_divergence": model_expected_prior_kl_divergence,
                "model_std_prior_kl_divergence": model_std_prior_kl_divergence,
                "model_inference_time": inference_time,
                "model_single_inference_time": single_inference_time,
                "model_posterior_expected_nll": model_posterior_expected_nll,
                "model_posterior_std_nll": model_posterior_std_nll
            })
            if "vi" in competitor_kwargs:
                _run.info.update({
                    "vi_posterior_kl_divergence": vi_posterior_kl_divergence,
                    "vi_time": vi_time,
                    "model_expected_elbo": model_expected_elbo,
                    "model_std_elbo": model_std_elbo,
                    "vi_expected_elbo": vi_expected_elbo,
                    "vi_std_elbo": vi_std_elbo,
                    "vi_expected_nll": vi_expected_nll,
                    "vi_std_nll": vi_std_nll
                })

        # Plotting
        if plot:
            assert torch.prod(torch.tensor(prior.event_shape)).item() == 1, \
                "plotting only supported for single output distributions"

            if bounds_func is None:
                def bounds_func(params_dict: dict[str, Tensor]) -> tuple[float, float]:
                    dist = complete_distribution.meta_prior.prior(**params_dict)
                    samples = dist.sample((10000,))
                    samples = samples.sort().values
                    return samples[499].item(), samples[9499].item()

            prior_plot = plot_distributions(prior, model_prior, None, model.sample_space_transform,
                                            bounds_func(phi_prior_dict), n_kl_samples=n_kl_samples)

            if len(competitor_kwargs) == 0:
                model_posterior_plot = plot_distributions(model_posterior, None, model.sample_space_transform,
                                                          n_kl_samples=n_kl_samples)

            if "vi" in competitor_kwargs:
                vi = GMMVI(model.n_components, model.state_size, prior, complete_distribution.observation_model,
                           inverse_transform)
                torch.set_grad_enabled(True)
                vi.fit(z, **competitor_kwargs["vi"])
                torch.set_grad_enabled(False)
                vi_posterior = vi.distribution()
                vi_posterior_plot = plot_distributions(vi_posterior, model_posterior, model.sample_space_transform,
                                                       model.sample_space_transform, bounds_func(phi_prior_dict),
                                                       n_kl_samples=None)

            if _run is not None:
                prior_plot.savefig(_run.observers[0].dir + "\\prior_plot.png")
                if len(competitor_kwargs) == 0:
                    model_posterior_plot.savefig(_run.observers[0].dir + "\\model_posterior_plot.png")
                if "vi" in competitor_kwargs:
                    vi_posterior_plot.savefig(_run.observers[0].dir + "\\vi_posterior_plot.png")
