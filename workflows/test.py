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

        print(f"GMM approximation prior mean KL divergence: {prior_kl_divergence}\n"
              f"Posterior mean KL divergence: {posterior_kl_divergence}\n")

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


            prior_plot = plot_distributions(exact_prior, model_prior, model.sample_space_transform,
                                            bounds_func(phi_prior_dict))

            posterior_plot = plot_distributions(exact_posterior, model_posterior, model.sample_space_transform,
                                                bounds_func(phi_posterior_dict))

            if _run is not None:
                prior_plot.savefig(_run.observers[0].dir + "\\prior_plot.png")
                posterior_plot.savefig(_run.observers[0].dir + "\\posterior_plot.png")
