"""
Testing workflow
"""

import torch
from torch import Tensor
from torch.nn import Identity
from torch.func import vmap, jacrev

from time import time
from typing import Callable, Optional, Union
from copy import copy

from model.distribution_transformer import DistributionTransformer
from distributions.distributions import CompleteDistribution, GaussianMixtureModel, ObservationModel
from distributions.utils import decode_gmm_sample, encode_gmm_sample, kl_divergence, plot_distributions, gmm_bounds_func
from competitor_methods.variational_inference import GMMVI
from competitor_methods.pfns import RiemannDistribution, PFN
from workflows.train import train_pfn
from workflows.utils import get_model_size
from dynamic.motion_models import LTIMotionModel
from dynamic.filters import LTIFilter
from dynamic.utils import plot_filtered_series


def test_conjugate_prior(model: DistributionTransformer,
                         complete_distribution: CompleteDistribution,
                         conjugacy_update: Callable[[dict[str, Tensor], dict[str, Tensor], str], dict[str, Tensor]],
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
    Standard testing routine for experiments involving conjugate priors

    Args:
        model: Model to Test.
        complete_distribution: Complete distribution over priors, state and observation.
        conjugacy_update: Function taking a dict of prior parameters, a dict of observations, and returning a dict of
            posterior parameters.
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
        device = gpu_device if torch.cuda.is_available() else 'cpu:0'
        scale_parametrisation = model.component_embedding.scale_parametrisation

        # Test inputs
        phi, x, z = complete_distribution.sample((n_test_priors,))

        # Device
        phi = phi.to(device)
        z = {key: val.to(device) for key, val in z.items()}
        model = model.to(device)

        # Exact solution
        phi_prior_dict = complete_distribution.meta_prior.decode_sample(phi)
        prior = complete_distribution.meta_prior.prior(**phi_prior_dict)
        phi_posterior_dict = conjugacy_update(phi_prior_dict, z, device)
        exact_posterior = complete_distribution.meta_prior.prior(**phi_posterior_dict)

        # Model solution
        start_time = time()
        phi_in, phi_out = model(phi.to(device), **z)
        model_inference_time = time() - start_time
        model_prior = GaussianMixtureModel(**decode_gmm_sample(phi_in, scale_parametrisation))
        model_posterior = GaussianMixtureModel(**decode_gmm_sample(phi_out, scale_parametrisation))

        model_prior_kl_divergence = kl_divergence(prior, model_prior, model.sample_space_transform,
                                                  n_kl_samples)
        model_expected_prior_kl_divergence = model_prior_kl_divergence.mean().item()
        model_std_prior_kl_divergence = model_prior_kl_divergence.std().item()
        model_posterior_kl_divergence = kl_divergence(exact_posterior, model_posterior, model.sample_space_transform,
                                                      n_kl_samples)
        model_expected_posterior_kl_divergence = model_posterior_kl_divergence.mean().item()
        model_std_posterior_kl_divergence = model_posterior_kl_divergence.std().item()

        model_posterior_nll = -model_posterior.log_prob(model.sample_space_transform(x)
                                                        .reshape(model_posterior.batch_shape
                                                                 + model_posterior.event_shape).to(device))
        model_posterior_nll -= torch.logdet(vmap(jacrev(model.sample_space_transform))
                                            (x.reshape(model_posterior.batch_shape).to(device)
                                             ).reshape(*prior.batch_shape, 1, 1))
        model_posterior_expected_nll = model_posterior_nll.mean().item()
        model_posterior_std_nll = model_posterior_nll.std().item()



        model_size = get_model_size(model)

        print(f"GMM approximation prior mean KL divergence: {model_expected_prior_kl_divergence}\n"
              f"Posterior mean KL divergence: {model_expected_posterior_kl_divergence}\n")

        if "vi" in competitor_kwargs:
            # VI solution
            before_vi = time()
            vi = GMMVI(model.n_components, model.state_size, prior, complete_distribution.observation_model,
                       inverse_transform, **competitor_kwargs["vi"]).to(device)
            torch.set_grad_enabled(True)
            vi.fit(z, **competitor_kwargs["vi"])
            torch.set_grad_enabled(False)
            vi_time = time() - before_vi
            vi_posterior_kl_divergence = kl_divergence(exact_posterior, vi.distribution(),
                                                       model.sample_space_transform,
                                                       n_kl_samples)
            vi_expected_kl_divergence = vi_posterior_kl_divergence.mean().item()
            vi_std_kl_divergence = vi_posterior_kl_divergence.std().item()
            vi_nll = -vi.distribution().log_prob(model.sample_space_transform(x)
                                                 .reshape(model_posterior.batch_shape
                                                          + model_posterior.event_shape).to(device))
            vi_nll -= torch.logdet(vmap(jacrev(model.sample_space_transform))
                                            (x.reshape(model_posterior.batch_shape).to(device)
                                             ).reshape(*prior.batch_shape, 1, 1))
            vi_expected_nll = vi_nll.mean().item()
            vi_std_nll = vi_nll.std().item()
            vi_elbo = -vi.posterior_loss(z, n_samples=n_test_priors)
            vi_expected_elbo = vi_elbo.mean().item()
            vi_std_elbo = vi_elbo.std().item()
            model_elbo = -vi.posterior_loss(z, n_samples=n_test_priors, distribution=model_posterior)
            model_expected_elbo = model_elbo.mean().item()
            model_std_elbo = model_elbo.std().item()

        if "pfns" in competitor_kwargs:
            assert torch.prod(torch.tensor(prior.event_shape)).item() == 1, \
                "pfns only supported for univariate output distributions"
            # PFN solution
            pfn_kwargs = copy(competitor_kwargs["pfns"])
            del pfn_kwargs["training_kwargs"]
            pfn = PFN(**pfn_kwargs, **model.observation_embeddings)
            torch.set_grad_enabled(True)
            pfn, _ = train_pfn(pfn, complete_distribution, _run=_run, **competitor_kwargs["pfns"]["training_kwargs"])
            torch.set_grad_enabled(False)
            pfn = pfn.to(device)
            start_time = time()
            phi_out = pfn(**z)
            pfn_inference_time = time() - start_time
            pfn_posterior = RiemannDistribution(phi_out, pfn.borders, pfn.infinite_support)

            pfn_posterior_kl_divergence = kl_divergence(exact_posterior, pfn_posterior,
                                                        None,
                                                        n_kl_samples)
            pfn_expected_kl_divergence = pfn_posterior_kl_divergence.mean().item()
            pfn_std_kl_divergence = pfn_posterior_kl_divergence.std().item()
            pfn_nll = -pfn_posterior.log_prob(x.reshape(pfn_posterior.batch_shape).to(device))
            pfn_expected_nll = pfn_nll.mean().item()
            pfn_std_nll = pfn_nll.std().item()

            pfn_size = get_model_size(pfn)

            if "vi" in competitor_kwargs:
                pfn_elbo = -vi.posterior_loss(z, n_samples=n_test_priors, distribution=pfn_posterior,
                                              inverse_transform=Identity())
                pfn_expected_elbo = pfn_elbo.mean().item()
                pfn_std_elbo = pfn_elbo.std().item()

        # Single problem run

        # Test inputs
        phi, x, z = complete_distribution.sample()

        # Device
        model.cpu()

        # Exact solution
        phi_prior_dict = complete_distribution.meta_prior.decode_sample(phi)
        prior = complete_distribution.meta_prior.prior(**phi_prior_dict)
        phi_posterior_dict = conjugacy_update(phi_prior_dict, z, "cpu")
        exact_posterior = complete_distribution.meta_prior.prior(**phi_posterior_dict)

        # Model solution
        start_time = time()
        phi_in, phi_out = model(phi, **z)
        model_single_inference_time = time() - start_time
        model_prior = GaussianMixtureModel(**decode_gmm_sample(phi_in, scale_parametrisation))
        model_posterior = GaussianMixtureModel(**decode_gmm_sample(phi_out, scale_parametrisation))

        if _run is not None:
            _run.info.update({
                "model_expected_prior_kl_divergence": model_expected_prior_kl_divergence,
                "model_std_prior_kl_divergence": model_std_prior_kl_divergence,
                "model_expected_posterior_kl_divergence": model_expected_posterior_kl_divergence,
                "model_std_posterior_kl_divergence": model_std_posterior_kl_divergence,
                "model_posterior_expected_nll": model_posterior_expected_nll,
                "model_posterior_std_nll": model_posterior_std_nll,
                "model_inference_time": model_inference_time,
                "model_single_inference_time": model_single_inference_time,
                "model_size": model_size
            })
        if "vi" in competitor_kwargs:
            _run.info.update({
                "vi_time": vi_time,
                "model_expected_elbo": model_expected_elbo,
                "model_std_elbo": model_std_elbo,
                "vi_expected_kl_divergence": vi_expected_kl_divergence,
                "vi_std_kl_divergence": vi_std_kl_divergence,
                "vi_expected_elbo": vi_expected_elbo,
                "vi_std_elbo": vi_std_elbo,
                "vi_expected_nll": vi_expected_nll,
                "vi_std_nll": vi_std_nll
            })
        if "pfns" in competitor_kwargs:
            _run.info.update({
                "pfn_inference_time": pfn_inference_time,
                "pfn_expected_kl_divergence": pfn_expected_kl_divergence,
                "pfn_std_kl_divergence": pfn_std_kl_divergence,
                "pfn_expected_nll": pfn_expected_nll,
                "pfn_std_nll": pfn_std_nll,
                "pfn_size": pfn_size
            })
            if "vi" in competitor_kwargs:
                _run.info.update({
                    "pfn_expected_elbo": pfn_expected_elbo,
                    "pfn_std_elbo": pfn_std_elbo,
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
                                            bounds_func(phi_prior_dict),
                                            n_kl_samples=n_kl_samples)

            posterior_plot = plot_distributions(exact_posterior, model_posterior, None, model.sample_space_transform,
                                                bounds_func(phi_posterior_dict),
                                                n_kl_samples=n_kl_samples)

            if "vi" in competitor_kwargs:
                vi = GMMVI(model.n_components, model.state_size, prior, complete_distribution.observation_model,
                           inverse_transform)
                torch.set_grad_enabled(True)
                start_time = time()
                vi.fit(z, **competitor_kwargs["vi"])
                vi_single_time = time() - start_time
                torch.set_grad_enabled(False)
                vi_posterior = vi.distribution()
                vi_posterior_plot = plot_distributions(exact_posterior, vi_posterior, None,
                                                       model.sample_space_transform, bounds_func(phi_posterior_dict))

            if "pfns" in competitor_kwargs:
                pfn = pfn.cpu()
                start_time = time()
                phi_out = pfn(**z)
                pfn_single_inference_time = time() - start_time
                pfn_posterior = RiemannDistribution(phi_out, pfn.borders, pfn.infinite_support)
                pfn_posterior_plot = plot_distributions(exact_posterior, pfn_posterior, None,
                                                        None, bounds_func(phi_posterior_dict))

            if _run is not None:
                prior_plot.savefig(_run.observers[0].dir + "\\prior_plot.png")
                posterior_plot.savefig(_run.observers[0].dir + "\\posterior_plot.png")
                if "vi" in competitor_kwargs:
                    _run.info.update({
                        "vi_single_time": vi_single_time
                    })
                    vi_posterior_plot.savefig(_run.observers[0].dir + "\\vi_posterior_plot.png")

                if "pfns" in competitor_kwargs:
                    pfn_posterior_plot.savefig(_run.observers[0].dir + "\\pfn_posterior_plot.png")
                    _run.info.update({
                        "pfn_single_inference_time": pfn_single_inference_time
                    })


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

        scale_parametrisation = model.component_embedding.scale_parametrisation

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
        model_inference_time = time() - start_time
        model_prior = GaussianMixtureModel(**decode_gmm_sample(phi_in, scale_parametrisation))
        model_posterior = GaussianMixtureModel(**decode_gmm_sample(phi_out, scale_parametrisation))

        prior_kl_divergence = kl_divergence(prior, model_prior, model.sample_space_transform,
                                            n_kl_samples)
        model_expected_prior_kl_divergence = prior_kl_divergence.mean().item()
        model_std_prior_kl_divergence = prior_kl_divergence.std().item()

        model_posterior_nll = -model_posterior.log_prob(model.sample_space_transform(x)
                                                        .reshape(model_posterior.batch_shape
                                                                 + model_posterior.event_shape).to(device))
        model_posterior_nll -= torch.logdet(vmap(jacrev(model.sample_space_transform))
                                            (x.reshape(model_posterior.batch_shape).to(device)
                                             ).reshape(*prior.batch_shape, 1, 1))
        model_posterior_expected_nll = model_posterior_nll.mean().item()
        model_posterior_std_nll = model_posterior_nll.std().item()

        model_size = get_model_size(model)

        print(f"GMM approximation prior mean KL divergence: {prior_kl_divergence.mean().item()}")

        if "vi" in competitor_kwargs:
            # VI solution
            before_vi = time()
            vi = GMMVI(model.n_components, model.state_size, prior, complete_distribution.observation_model,
                       inverse_transform, **competitor_kwargs["vi"]).to(device)
            torch.set_grad_enabled(True)
            vi.fit(z, **competitor_kwargs["vi"])
            torch.set_grad_enabled(False)
            vi_time = time() - before_vi
            vi_nll = -vi.distribution().log_prob(model.sample_space_transform(x)
                                                 .reshape(model_posterior.batch_shape
                                                          + model_posterior.event_shape).to(device))
            vi_nll -= torch.logdet(vmap(jacrev(model.sample_space_transform))
                                   (x.reshape(model_posterior.batch_shape).to(device)
                                    ).reshape(*prior.batch_shape, 1, 1))
            vi_expected_nll = vi_nll.mean().item()
            vi_std_nll = vi_nll.std().item()
            vi_elbo = -vi.posterior_loss(z, n_samples=n_test_priors)
            vi_expected_elbo = vi_elbo.mean().item()
            vi_std_elbo = vi_elbo.std().item()
            model_elbo = -vi.posterior_loss(z, n_samples=n_test_priors, distribution=model_posterior)
            model_expected_elbo = model_elbo.mean().item()
            model_std_elbo = model_elbo.std().item()

        if "pfns" in competitor_kwargs:
            assert torch.prod(torch.tensor(prior.event_shape)).item() == 1, \
                "pfns only supported for univariate output distributions"
            # PFN solution
            pfn_kwargs = copy(competitor_kwargs["pfns"])
            del pfn_kwargs["training_kwargs"]
            pfn = PFN(**pfn_kwargs, **model.observation_embeddings)
            torch.set_grad_enabled(True)
            pfn, _ = train_pfn(pfn, complete_distribution, _run=_run, **competitor_kwargs["pfns"]["training_kwargs"])
            torch.set_grad_enabled(False)
            pfn = pfn.to(device)
            start_time = time()
            phi_out = pfn(**z)
            pfn_inference_time = time() - start_time
            pfn_posterior = RiemannDistribution(phi_out, pfn.borders, pfn.infinite_support)

            pfn_nll = -pfn_posterior.log_prob(x.reshape(pfn_posterior.batch_shape).to(device))
            pfn_expected_nll = pfn_nll.mean().item()
            pfn_std_nll = pfn_nll.std().item()

            pfn_size = get_model_size(pfn)

            if "vi" in competitor_kwargs:
                pfn_elbo = -vi.posterior_loss(z, n_samples=n_test_priors, distribution=pfn_posterior,
                                              inverse_transform=Identity())
                pfn_expected_elbo = pfn_elbo.mean().item()
                pfn_std_elbo = pfn_elbo.std().item()

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
        model_single_inference_time = time() - start_time
        model_prior = GaussianMixtureModel(**decode_gmm_sample(phi_in, scale_parametrisation))
        model_posterior = GaussianMixtureModel(**decode_gmm_sample(phi_out, scale_parametrisation))

        if _run is not None:
            _run.info.update({
                "model_expected_prior_kl_divergence": model_expected_prior_kl_divergence,
                "model_std_prior_kl_divergence": model_std_prior_kl_divergence,
                "model_inference_time": model_inference_time,
                "model_single_inference_time": model_single_inference_time,
                "model_posterior_expected_nll": model_posterior_expected_nll,
                "model_posterior_std_nll": model_posterior_std_nll,
                "model_size": model_size,
            })
            if "vi" in competitor_kwargs:
                _run.info.update({
                    "vi_time": vi_time,
                    "model_expected_elbo": model_expected_elbo,
                    "model_std_elbo": model_std_elbo,
                    "vi_expected_elbo": vi_expected_elbo,
                    "vi_std_elbo": vi_std_elbo,
                    "vi_expected_nll": vi_expected_nll,
                    "vi_std_nll": vi_std_nll
                })
            if "pfns" in competitor_kwargs:
                _run.info.update({
                    "pfn_inference_time": pfn_inference_time,
                    "pfn_expected_nll": pfn_expected_nll,
                    "pfn_std_nll": pfn_std_nll,
                    "pfn_size": pfn_size
                })
                if "vi" in competitor_kwargs:
                    _run.info.update({
                        "pfn_expected_elbo": pfn_expected_elbo,
                        "pfn_std_elbo": pfn_std_elbo,
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
                start_time = time()
                vi.fit(z, **competitor_kwargs["vi"])
                vi_single_time = time() - start_time
                torch.set_grad_enabled(False)
                vi_posterior = vi.distribution()
                vi_posterior_plot = plot_distributions(vi_posterior, model_posterior, model.sample_space_transform,
                                                       model.sample_space_transform, bounds_func(phi_prior_dict),
                                                       n_kl_samples=None)

            if "pfns" in competitor_kwargs:
                pfn = pfn.cpu()
                start_time = time()
                phi_out = pfn(**z)
                pfn_single_inference_time = time() - start_time
                pfn_posterior = RiemannDistribution(phi_out, pfn.borders, pfn.infinite_support)
                pfn_posterior_plot = plot_distributions(pfn_posterior, model_posterior, None,
                                                        model.sample_space_transform, bounds_func(phi_prior_dict),
                                                        n_kl_samples=None)

            if _run is not None:
                prior_plot.savefig(_run.observers[0].dir + "\\prior_plot.png")
                if len(competitor_kwargs) == 0:
                    model_posterior_plot.savefig(_run.observers[0].dir + "\\model_posterior_plot.png")
                if "vi" in competitor_kwargs:
                    _run.info.update({
                        "vi_single_time": vi_single_time
                    })
                    vi_posterior_plot.savefig(_run.observers[0].dir + "\\vi_posterior_plot.png")

                if "pfns" in competitor_kwargs:
                    pfn_posterior_plot.savefig(_run.observers[0].dir + "\\pfn_posterior_plot.png")
                    _run.info.update({
                        "pfn_single_inference_time": pfn_single_inference_time
                    })


def test_lti_filter(model: DistributionTransformer,
                    motion_model: LTIMotionModel,
                    observation_model: dict[str, ObservationModel],
                    competitor_kwargs: Optional[dict[str, dict]] = None,
                    series_length: int = 1000,
                    n_test_series: int = 1000,
                    plotting_kwargs: Optional[dict] = None,
                    gpu_device: str = "cuda:0",
                    _run=None
                    ) -> None:
    """
    Test model on Bayesian filtering task. Currently restricted to GMM initial priors.

    Args:
        model: Model to test.
        motion_model: Motion model for dynamical system.
        observation_model: Observation model for dynamical system.
        competitor_kwargs: Kwargs for competitor methods.
            Defaults to None.
        series_length: Length of series to test on.
            Defaults to 1000.
        n_test_series: Number of series to test on.
            Defaults to 1000.
        plotting_kwargs: Plotting kwargs. Set to None to disable plotting.
            Defaults to NOne.
        gpu_device: GPU device to test on.
            Defaults to cuda:0.
        _run: Sacred run object.

    Returns:

    """
    with torch.no_grad():
        device = gpu_device if torch.cuda.is_available() else 'cpu:0'
        model.to(device)
        scale_parametrisation = model.component_embedding.scale_parametrisation

        model_size = get_model_size(model)

        series = motion_model.sample((series_length, n_test_series))

        for obs_model in observation_model.values():
            obs_model.condition_(series)

        observation_series = {key: obs_model.sample().to(device) for key, obs_model in observation_model.items()}

        # Model solution
        filter = LTIFilter(model, motion_model)

        start_time = time()
        filtered_series = filter.filter(observation_series, motion_model.x0_distribution)
        model_inference_time = time() - start_time

        filtered_series_dict = decode_gmm_sample(filtered_series, scale_parametrisation)
        model_nll = -GaussianMixtureModel(**filtered_series_dict).log_prob(series)
        model_expected_nll = model_nll.mean().item()
        model_std_nll = model_nll.std().item()

        # Single problem run

        # Device
        model.cpu()

        series = motion_model.sample((series_length,))

        for obs_model in observation_model.values():
            obs_model.condition_(series)

        observation_series = {key: obs_model.sample().cpu() for key, obs_model in observation_model.items()}

        # Model solution
        filter = LTIFilter(model, motion_model)

        start_time = time()
        filtered_series = filter.filter(observation_series, motion_model.x0_distribution)
        model_single_inference_time = time() - start_time

        if _run is not None:
            _run.info.update({
                "model_inference_time": model_inference_time,
                "model_single_inference_time": model_single_inference_time,
                "model_expected_nll": model_expected_nll,
                "model_std_nll": model_std_nll,
                "model_size": model_size
            })

        # Plotting first dimension of state space
        if plotting_kwargs is not None:
            # Select dimension
            dim = plotting_kwargs["dim"]

            filtered_series_dict = decode_gmm_sample(filtered_series, scale_parametrisation)
            filtered_series_dict["loc"] = filtered_series_dict["loc"][..., dim].unsqueeze(-1)
            filtered_series_dict[scale_parametrisation] = \
                filtered_series_dict[scale_parametrisation].diagonal(dim1=-2, dim2=-1)[..., dim].unsqueeze(
                    -1).unsqueeze(-1)
            filtered_series = encode_gmm_sample(filtered_series_dict, scale_parametrisation)

            bounds = list(zip(*[gmm_bounds_func(decode_gmm_sample(dist, scale_parametrisation))
                                for dist in filtered_series]))
            bounds = (max(bounds[0]), min(bounds[1]))
            bounds = (max(bounds[0], series.max().item() + 1), min(bounds[1], series.min().item() - 1))

            filter_distribution = GaussianMixtureModel(**filtered_series_dict)

            model_series_plot = plot_filtered_series(filter_distribution, series, bounds, **plotting_kwargs)

            if _run is not None:
                model_series_plot.savefig(_run.observers[0].dir + "\\model_series_plot.png")
