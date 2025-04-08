"""
Testing workflow
"""
import torch
from torch import Tensor
from torch.nn import Identity
from torch.func import vmap, jacrev
from torch.distributions import MultivariateNormal

from time import time
from math import sqrt
from typing import Callable, Optional, Union
from copy import copy
import matplotlib.pyplot as plt
import numpy as np

from model.distribution_transformer import DistributionTransformer
from distributions.distributions import CompleteDistribution, GaussianMixtureModel, ObservationModel
from distributions.utils import decode_gmm_sample, encode_gmm_sample, kl_divergence, plot_distributions
from competitor_methods.variational_inference import VI
from competitor_methods.pfns import RiemannDistribution, PFN
from competitor_methods.ekf import EKF
from competitor_methods.particle_filter import ParticleFilter
from workflows.train import train_pfn
from workflows.utils import get_model_size
from dynamic.motion_models import LTIMotionModel
from dynamic.filters import LTIFilter
from dynamic.utils import plot_filtered_series


def test_conjugate_prior(model_list: Union[DistributionTransformer, list[DistributionTransformer]],
                         n_components_list: Union[list[int], int],
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
        model_list: Model to Test.
        n_components_list: Number of GMM components
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
    if isinstance(model_list, DistributionTransformer):
        model_list = [model_list]

    if isinstance(n_components_list, int):
        n_components_list = [n_components_list]

    with torch.no_grad():
        device = gpu_device if torch.cuda.is_available() else 'cpu:0'
        scale_parametrisation = model_list[0].component_embedding.scale_parametrisation

        # Test inputs
        phi, x, z = complete_distribution.sample((n_test_priors,))

        # Device
        phi = phi.to(device)
        z = {key: val.to(device) for key, val in z.items()}

        # Exact solution
        phi_prior_dict = complete_distribution.meta_prior.decode_sample(phi)
        prior = complete_distribution.meta_prior.prior(**phi_prior_dict)
        phi_posterior_dict = conjugacy_update(phi_prior_dict, z, device)
        exact_posterior = complete_distribution.meta_prior.prior(**phi_posterior_dict)

        # Model solution
        model_inference_time = []
        model_expected_prior_kl_divergence = []
        model_conf_prior_kl_divergence = []
        model_expected_posterior_kl_divergence = []
        model_conf_posterior_kl_divergence = []
        model_posterior_expected_nll = []
        model_posterior_conf_nll = []
        model_size = []

        for model, n_components in zip(model_list, n_components_list):
            model = model.to(device)
            start_time = time()
            phi_in, phi_out = model(phi.to(device), **z)
            model_inference_time.append(time() - start_time)
            model_prior = GaussianMixtureModel(**decode_gmm_sample(phi_in, scale_parametrisation))
            model_posterior = GaussianMixtureModel(**decode_gmm_sample(phi_out, scale_parametrisation))

            model_prior_kl_divergence = kl_divergence(prior, model_prior, model.sample_space_transform,
                                                      n_kl_samples)
            model_expected_prior_kl_divergence.append(model_prior_kl_divergence.mean().item())
            model_conf_prior_kl_divergence.append(model_prior_kl_divergence.std().item() * 1.96 / sqrt(n_test_priors))
            model_posterior_kl_divergence = kl_divergence(exact_posterior, model_posterior, model.sample_space_transform,
                                                          n_kl_samples)
            model_expected_posterior_kl_divergence.append(model_posterior_kl_divergence.mean().item())
            model_conf_posterior_kl_divergence.append(model_posterior_kl_divergence.std().item() * 1.96 / sqrt(n_test_priors))

            event_shape = model_posterior.event_shape
            event_shape = torch.Size([1]) if event_shape == torch.Size([]) else event_shape
            model_posterior_nll = -model_posterior.log_prob(model.sample_space_transform(x)
                                                            .reshape(model_posterior.batch_shape
                                                                     + event_shape).to(device))
            model_posterior_nll -= torch.logdet(vmap(jacrev(model.sample_space_transform))
                                                (x.reshape(model_posterior.batch_shape + model_posterior.event_shape
                                                           ).to(device)
                                                 ).reshape(prior.batch_shape + event_shape + event_shape))
            model_posterior_expected_nll.append(model_posterior_nll.mean().item())
            model_posterior_conf_nll.append(model_posterior_nll.std().item() * 1.96 / sqrt(n_test_priors))

            model_size.append(get_model_size(model))

            print(f"{n_components}-component GMM approximation prior mean KL divergence: {model_prior_kl_divergence.mean().item()}\n"
                  f"{n_components}-component posterior mean KL divergence: {model_posterior_kl_divergence.mean().item()}\n")

        if "vi" in competitor_kwargs:
            # VI solution
            vi = VI(model_list[0].state_size, prior, complete_distribution.observation_model, inverse_transform,
                    **competitor_kwargs["vi"]).to(device)
            vi_expected_nll_series = []
            vi_conf_nll_series = []
            vi_time_series = []
            if "repeats" in competitor_kwargs["vi"]:
                repeats = competitor_kwargs["vi"]["repeats"]
            else:
                repeats = 1

            for i in range(repeats):
                before_vi = time()
                torch.set_grad_enabled(True)
                vi.fit(z, epoch=i, num_epochs=repeats, **competitor_kwargs["vi"])
                torch.set_grad_enabled(False)
                vi_time = time() - before_vi
                vi_nll = -vi.distribution().log_prob(model_list[0].sample_space_transform(x)
                                                     .reshape(model_posterior.batch_shape
                                                              + model_posterior.event_shape).to(device))
                vi_nll -= torch.logdet(vmap(jacrev(model_list[0].sample_space_transform))
                                       (x.reshape(model_posterior.batch_shape
                                                  + model_posterior.event_shape).to(device)
                                        ).reshape(prior.batch_shape + model_posterior.event_shape
                                                  + model_posterior.event_shape))
                vi_expected_nll = vi_nll.mean().item()
                vi_conf_nll = vi_nll.std().item() * 1.96 / sqrt(n_test_priors)
                vi_expected_nll_series.append(vi_expected_nll)
                vi_conf_nll_series.append(vi_conf_nll)
                vi_time_series.append(vi_time)
            vi_time = sum(vi_time_series)
            vi_time_series = np.array(vi_time_series).cumsum()
            vi_posterior_kl_divergence = kl_divergence(exact_posterior, vi.distribution(),
                                                       model_list[0].sample_space_transform,
                                                       n_kl_samples)
            vi_expected_kl_divergence = vi_posterior_kl_divergence.mean().item()
            vi_conf_kl_divergence = vi_posterior_kl_divergence.std().item() * 1.96 / sqrt(n_test_priors)
            vi_elbo = -vi.posterior_loss(z, n_samples=n_test_priors)
            vi_expected_elbo = vi_elbo.mean().item()
            vi_conf_elbo = vi_elbo.std().item() * 1.96 / sqrt(n_test_priors)
            model_elbo = -vi.posterior_loss(z, n_samples=n_test_priors, distribution=model_posterior)
            model_expected_elbo = model_elbo.mean().item()
            model_conf_elbo = model_elbo.std().item() * 1.96 / sqrt(n_test_priors)

        if "pfns" in competitor_kwargs:
            assert torch.prod(torch.tensor(prior.event_shape)).item() == 1, \
                "pfns only supported for univariate output distributions"
            # PFN solution
            pfn_kwargs = copy(competitor_kwargs["pfns"])
            del pfn_kwargs["training_kwargs"]
            del pfn_kwargs["n_buckets"]

            if isinstance(competitor_kwargs["pfns"]["n_buckets"], list):
                n_buckets_list = competitor_kwargs["pfns"]["n_buckets"]
            else:
                n_buckets_list = [competitor_kwargs["pfns"]["n_buckets"]]

            pfns = []
            pfn_inference_time = []
            pfn_expected_kl_divergence = []
            pfn_conf_kl_divergence = []
            pfn_expected_nll = []
            pfn_conf_nll = []
            pfn_size = []

            for n_buckets in n_buckets_list:
                pfn = PFN(n_buckets=n_buckets, **pfn_kwargs, **copy(model_list[0].observation_embeddings))
                torch.set_grad_enabled(True)
                pfn, _ = train_pfn(pfn, complete_distribution, _run=_run, **competitor_kwargs["pfns"]["training_kwargs"])
                torch.set_grad_enabled(False)
                pfn = pfn.to(device)
                start_time = time()
                phi_out = pfn(**z)
                pfns.append(pfn)
                pfn_inference_time.append(time() - start_time)
                pfn_posterior = RiemannDistribution(phi_out, pfn.borders, pfn.infinite_support)

                pfn_posterior_kl_divergence = kl_divergence(exact_posterior, pfn_posterior,
                                                            None,
                                                            n_kl_samples)
                pfn_expected_kl_divergence.append(pfn_posterior_kl_divergence.mean().item())
                pfn_conf_kl_divergence.append(pfn_posterior_kl_divergence.std().item() * 1.96 / sqrt(n_test_priors))
                pfn_nll = -pfn_posterior.log_prob(x.reshape(pfn_posterior.batch_shape).to(device))
                pfn_expected_nll.append(pfn_nll.mean().item())
                pfn_conf_nll.append(pfn_nll.std().item() * 1.96 / sqrt(n_test_priors))

                pfn_size.append(get_model_size(pfn))
                """
                if "vi" in competitor_kwargs:
                    pfn_elbo = -vi.posterior_loss(z, n_samples=n_test_priors, distribution=pfn_posterior,
                                                  inverse_transform=Identity())
                    pfn_expected_elbo = pfn_elbo.mean().item()
                    pfn_conf_elbo = pfn_elbo.std().item() * 1.96 / sqrt(n_test_priors)
                """

        """
        if "vi" in competitor_kwargs:
            # Plot loss timeseries
            plt.style.use(['seaborn-v0_8-paper'])
            fig, ax = plt.subplots()
            ax.plot(vi_time_series, vi_expected_nll_series,
                    "-", color="tab:orange")
            if "pfns" in competitor_kwargs:
                ax.plot([pfn_inference_time, vi_time_series[-1]], [pfn_expected_nll] * 2,
                        "--", color="tab:green")
            ax.plot([model_inference_time, vi_time_series[-1]], [model_posterior_expected_nll] * 2,
                    "--", color="tab:blue")
            ax.legend(["SVI"] + ["PFN"] * ("pfns" in competitor_kwargs) + ["Distribution Transformer"])
            ax.fill_between(vi_time_series, np.array(vi_expected_nll_series) + np.array(vi_conf_nll_series),
                            np.array(vi_expected_nll_series) - np.array(vi_conf_nll_series),
                            color="tab:orange", alpha=0.5)
            if "pfns" in competitor_kwargs:
                ax.fill_between([pfn_inference_time, vi_time_series[-1]],
                                np.array([pfn_expected_nll] * 2) + np.array([pfn_conf_nll] * 2),
                                np.array([pfn_expected_nll] * 2) - np.array([pfn_conf_nll] * 2),
                                color="tab:green", alpha=0.5)
            ax.fill_between([model_inference_time, vi_time_series[-1]],
                            np.array([model_posterior_expected_nll] * 2) + np.array([model_posterior_conf_nll] * 2),
                            np.array([model_posterior_expected_nll] * 2) - np.array([model_posterior_conf_nll] * 2),
                            color="tab:blue", alpha=0.5)
            ax.set_xlabel(f"Inference Time per {n_test_priors} task batch (s)")
            ax.set_ylabel("Negative Log-Likelihood")
            ax.set_xscale("log")
            fig.savefig(_run.observers[0].dir + "\\loss_series.pdf", format="pdf")
            plt.show()
        """

        # Single problem run

        # Test inputs
        phi, x, z = complete_distribution.sample()

        # Exact solution
        phi_prior_dict = complete_distribution.meta_prior.decode_sample(phi)
        prior = complete_distribution.meta_prior.prior(**phi_prior_dict)
        phi_posterior_dict = conjugacy_update(phi_prior_dict, z, "cpu")
        exact_posterior = complete_distribution.meta_prior.prior(**phi_posterior_dict)

        model_single_inference_time = []
        model_prior = []
        model_posterior = []
        for model, n_components in zip(model_list, n_components_list):
            # Model solution

            # Device
            model.cpu()

            start_time = time()
            phi_in, phi_out = model(phi, **z)
            model_single_inference_time.append(time() - start_time)
            model_prior.append(GaussianMixtureModel(**decode_gmm_sample(phi_in, scale_parametrisation)))
            model_posterior.append(GaussianMixtureModel(**decode_gmm_sample(phi_out, scale_parametrisation)))

        if _run is not None:
            _run.info.update({
                f"model_expected_prior_kl_divergence": model_expected_prior_kl_divergence,
                f"model_conf_prior_kl_divergence": model_conf_prior_kl_divergence,
                f"model_expected_posterior_kl_divergence": model_expected_posterior_kl_divergence,
                f"model_conf_posterior_kl_divergence": model_conf_posterior_kl_divergence,
                f"model_posterior_expected_nll": model_posterior_expected_nll,
                f"model_posterior_conf_nll": model_posterior_conf_nll,
                f"model_inference_time": model_inference_time,
                f"model_single_inference_time": model_single_inference_time,
                f"model_size": model_size
            })
        if "vi" in competitor_kwargs:
            _run.info.update({
                "vi_time": vi_time,
                f"model_expected_elbo": model_expected_elbo,
                f"model_conf_elbo": model_conf_elbo,
                "vi_expected_kl_divergence": vi_expected_kl_divergence,
                "vi_conf_kl_divergence": vi_conf_kl_divergence,
                "vi_expected_elbo": vi_expected_elbo,
                "vi_conf_elbo": vi_conf_elbo,
                "vi_expected_nll": vi_expected_nll,
                "vi_conf_nll": vi_conf_nll
            })
        if "pfns" in competitor_kwargs:
            _run.info.update({
                "pfn_inference_time": pfn_inference_time,
                "pfn_expected_kl_divergence": pfn_expected_kl_divergence,
                "pfn_conf_kl_divergence": pfn_conf_kl_divergence,
                "pfn_expected_nll": pfn_expected_nll,
                "pfn_conf_nll": pfn_conf_nll,
                "pfn_size": pfn_size
            })
            """
            if "vi" in competitor_kwargs:
                _run.info.update({
                    "pfn_expected_elbo": pfn_expected_elbo,
                    "pfn_conf_elbo": pfn_conf_elbo,
                })
            """

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


            prior_plot = []
            posterior_plot = []
            for q, n_components in zip(model_prior, n_components_list):
                prior_plot.append(plot_distributions(prior, q, None, model_list[0].sample_space_transform,
                                                     bounds_func(phi_prior_dict), n_kl_samples=None,
                                                     legend=["Exact Prior",
                                                             f"{n_components}-component Distribution Transformer"]))

            for q, n_components in zip(model_posterior, n_components_list):
                posterior_plot.append(plot_distributions(exact_posterior, q, None, model_list[0].sample_space_transform,
                                                         bounds_func(phi_posterior_dict), n_kl_samples=None,
                                                         legend=["Exact Posterior",
                                                                 f"{n_components}-component Distribution Transformer"]))

            if "vi" in competitor_kwargs:
                vi = VI(model_list[0].state_size, prior, complete_distribution.observation_model, inverse_transform)
                torch.set_grad_enabled(True)
                start_time = time()
                for _ in range(repeats):
                    vi.fit(z, **competitor_kwargs["vi"])
                vi_single_time = time() - start_time
                torch.set_grad_enabled(False)
                vi_posterior = vi.distribution()
                vi_posterior_plot = plot_distributions(exact_posterior, vi_posterior, None,
                                                       model_list[0].sample_space_transform,
                                                       bounds_func(phi_posterior_dict),
                                                       n_kl_samples=None, legend=["Exact Posterior", "SVI"])

            if "pfns" in competitor_kwargs:
                pfn_posteriors = []
                pfn_posterior_plots = []
                pfn_single_inference_time = []
                for n_buckets, pfn in zip(n_buckets_list, pfns):
                    pfn = pfn.cpu()
                    start_time = time()
                    phi_out = pfn(**z)
                    pfn_single_inference_time.append(time() - start_time)
                    pfn_posterior = RiemannDistribution(phi_out, pfn.borders, pfn.infinite_support)
                    pfn_posteriors.append(pfn_posterior)
                    pfn_posterior_plot = plot_distributions(exact_posterior, pfn_posterior, None,
                                                            None, bounds_func(phi_posterior_dict),
                                                            n_kl_samples=None, legend=["Exact Posterior",
                                                                                       f"{n_buckets}-bucket PFN"])
                    pfn_posterior_plots.append(pfn_posterior_plot)

                plt.style.use(['seaborn-v0_8-paper'])
                bounds = bounds_func(phi_posterior_dict)
                n_points = 1000
                q_transform = model_list[0].sample_space_transform

                points = torch.linspace(*bounds, steps=n_points)
                exact_density = torch.exp(exact_posterior.log_prob(points.reshape((n_points,)
                                                                                  + exact_posterior.event_shape)))

                fig, ax = plt.subplots()
                ax.plot(points, exact_density)

                for pfn_posterior in pfn_posteriors:
                    pfn_density = torch.exp(pfn_posterior.log_prob(points).reshape((n_points,)
                                                                                   + exact_posterior.event_shape))
                    ax.plot(points, pfn_density)
                for q in model_posterior:
                    ax.plot(points, torch.exp(q.log_prob(q_transform(points).reshape((n_points,) + q.event_shape))
                                              + torch.log(vmap(jacrev(q_transform))
                                                          (points).reshape((-1,) + exact_posterior.batch_shape))))

                ax.legend(["Exact"] + [f"{n_buckets}-bucket PFN" for n_buckets in n_buckets_list]
                          + [f"{n}-component DT" for n in n_components_list])
                ax.set_ylabel("Probability Density")
                ax.set_xlabel("Sample Space")
                plt.show()

            if _run is not None:
                fig.savefig(_run.observers[0].dir + f"\\combined_posterior_plot.pdf", format="pdf")
                for prior, posterior, n in zip(prior_plot, posterior_plot, n_components_list):
                    prior.savefig(_run.observers[0].dir + f"\\prior_plot_{n}.pdf", format="pdf")
                    posterior.savefig(_run.observers[0].dir + f"\\posterior_plot_{n}.pdf", format="pdf")
                if "vi" in competitor_kwargs:
                    _run.info.update({
                        "vi_single_time": vi_single_time
                    })
                    vi_posterior_plot.savefig(_run.observers[0].dir + "\\vi_posterior_plot.pdf", format="pdf")

                if "pfns" in competitor_kwargs:
                    for n_buckets, pfn_posterior_plot in zip(n_buckets_list, pfn_posterior_plots):
                        pfn_posterior_plot.savefig(_run.observers[0].dir + f"\\pfn_posterior_plot_{n_buckets}.pdf",
                                                   format="pdf")
                    _run.info.update({
                        "pfn_single_inference_time": pfn_single_inference_time
                    })


def test(model: DistributionTransformer,
         n_components: int,
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
        n_components: Number of GMM components.
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
        model_conf_prior_kl_divergence = prior_kl_divergence.std().item() * 1.96 / sqrt(n_test_priors)

        model_posterior_nll = -model_posterior.log_prob(model.sample_space_transform(x)
                                                        .reshape(model_posterior.batch_shape
                                                                 + model_posterior.event_shape).to(device))
        model_posterior_nll -= torch.logdet(vmap(jacrev(model.sample_space_transform))
                                            (x.reshape(model_posterior.batch_shape
                                                       + model_posterior.event_shape).to(device)
                                             ).reshape(prior.batch_shape + model_posterior.event_shape
                                                       + model_posterior.event_shape))
        model_posterior_expected_nll = model_posterior_nll.mean().item()
        model_posterior_conf_nll = model_posterior_nll.std().item() * 1.96 / sqrt(n_test_priors)

        model_size = get_model_size(model)

        print(f"GMM approximation prior mean KL divergence: {prior_kl_divergence.mean().item()}")

        if "vi" in competitor_kwargs:
            # VI solution
            vi = VI(model.state_size, prior, complete_distribution.observation_model, inverse_transform,
                    **competitor_kwargs["vi"]).to(device)
            vi_expected_nll_series = []
            vi_conf_nll_series = []
            vi_time_series = []
            if "repeats" in competitor_kwargs["vi"]:
                repeats = competitor_kwargs["vi"]["repeats"]
            else:
                repeats = 1

            for i in range(repeats):
                before_vi = time()
                torch.set_grad_enabled(True)
                vi.fit(z, epoch=i, num_epochs=repeats, **competitor_kwargs["vi"])
                torch.set_grad_enabled(False)
                vi_time = time() - before_vi
                vi_nll = -vi.distribution().log_prob(model.sample_space_transform(x)
                                                     .reshape(model_posterior.batch_shape
                                                              + model_posterior.event_shape).to(device))
                vi_nll -= torch.logdet(vmap(jacrev(model.sample_space_transform))
                                       (x.reshape(model_posterior.batch_shape
                                                  + model_posterior.event_shape).to(device)
                                        ).reshape(prior.batch_shape + model_posterior.event_shape
                                                  + model_posterior.event_shape))
                vi_expected_nll = vi_nll.mean().item()
                vi_conf_nll = vi_nll.std().item() * 1.96 / sqrt(n_test_priors)
                vi_expected_nll_series.append(vi_expected_nll)
                vi_conf_nll_series.append(vi_conf_nll)
                vi_time_series.append(vi_time)

            vi_time = sum(vi_time_series)
            vi_time_series = np.array(vi_time_series).cumsum()
            vi_elbo = -vi.posterior_loss(z, n_samples=n_test_priors)
            vi_expected_elbo = vi_elbo.mean().item()
            vi_conf_elbo = vi_elbo.std().item() * 1.96 / sqrt(n_test_priors)
            model_elbo = -vi.posterior_loss(z, n_samples=n_test_priors, distribution=model_posterior)
            model_expected_elbo = model_elbo.mean().item()
            model_conf_elbo = model_elbo.std().item() * 1.96 / sqrt(n_test_priors)

        if "pfns" in competitor_kwargs:
            assert torch.prod(torch.tensor(prior.event_shape)).item() == 1, \
                "pfns only supported for univariate output distributions"
            # PFN solution
            pfn_kwargs = copy(competitor_kwargs["pfns"])
            del pfn_kwargs["training_kwargs"]
            pfn = PFN(**pfn_kwargs, **copy(model.observation_embeddings))
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
            pfn_conf_nll = pfn_nll.std().item() * 1.96 / sqrt(n_test_priors)

            pfn_size = get_model_size(pfn)

            if "vi" in competitor_kwargs:
                pfn_elbo = -vi.posterior_loss(z, n_samples=n_test_priors, distribution=pfn_posterior,
                                              inverse_transform=Identity())
                pfn_expected_elbo = pfn_elbo.mean().item()
                pfn_conf_elbo = pfn_elbo.std().item() * 1.96 / sqrt(n_test_priors)

        if "vi" in competitor_kwargs:
            # Plot loss timeseries
            plt.style.use(['seaborn-v0_8-paper'])
            fig, ax = plt.subplots()
            ax.plot(vi_time_series, vi_expected_nll_series,
                    "-", color="tab:orange")
            if "pfns" in competitor_kwargs:
                ax.plot([pfn_inference_time, vi_time_series[-1]], [pfn_expected_nll] * 2,
                        "--", color="tab:green")
            ax.plot([model_inference_time, vi_time_series[-1]], [model_posterior_expected_nll] * 2,
                    "--", color="tab:blue")
            ax.legend(["SVI"] + ["PFN"] * ("pfns" in competitor_kwargs) + ["Distribution Transformer"])
            ax.fill_between(vi_time_series, np.array(vi_expected_nll_series) + np.array(vi_conf_nll_series),
                            np.array(vi_expected_nll_series) - np.array(vi_conf_nll_series),
                            color="tab:orange", alpha=0.5)
            if "pfns" in competitor_kwargs:
                ax.fill_between([pfn_inference_time, vi_time_series[-1]],
                                np.array([pfn_expected_nll] * 2) + np.array([pfn_conf_nll] * 2),
                                np.array([pfn_expected_nll] * 2) - np.array([pfn_conf_nll] * 2),
                                color="tab:green", alpha=0.5)
            ax.fill_between([model_inference_time, vi_time_series[-1]],
                            np.array([model_posterior_expected_nll] * 2) + np.array([model_posterior_conf_nll] * 2),
                            np.array([model_posterior_expected_nll] * 2) - np.array([model_posterior_conf_nll] * 2),
                            color="tab:blue", alpha=0.5)
            ax.set_xlabel(f"Inference Time per {n_test_priors} Problem Batch (s)")
            ax.set_ylabel("Negative Log-Likelihood")
            ax.set_xscale("log")
            fig.savefig(_run.observers[0].dir + "\\loss_series.pdf", format="pdf")
            plt.show()

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
                f"model_{n_components}_expected_prior_kl_divergence": model_expected_prior_kl_divergence,
                f"model_{n_components}_conf_prior_kl_divergence": model_conf_prior_kl_divergence,
                f"model_{n_components}_inference_time": model_inference_time,
                f"model_{n_components}_single_inference_time": model_single_inference_time,
                f"model_{n_components}_posterior_expected_nll": model_posterior_expected_nll,
                f"model_{n_components}_posterior_conf_nll": model_posterior_conf_nll,
                f"model_{n_components}_size": model_size,
            })
            if "vi" in competitor_kwargs:
                _run.info.update({
                    "vi_time": vi_time,
                    f"model_{n_components}_expected_elbo": model_expected_elbo,
                    f"model_{n_components}_conf_elbo": model_conf_elbo,
                    "vi_expected_elbo": vi_expected_elbo,
                    "vi_conf_elbo": vi_conf_elbo,
                    "vi_expected_nll": vi_expected_nll,
                    "vi_conf_nll": vi_conf_nll
                })
            if "pfns" in competitor_kwargs:
                _run.info.update({
                    "pfn_inference_time": pfn_inference_time,
                    "pfn_expected_nll": pfn_expected_nll,
                    "pfn_conf_nll": pfn_conf_nll,
                    "pfn_size": pfn_size
                })
                if "vi" in competitor_kwargs:
                    _run.info.update({
                        "pfn_expected_elbo": pfn_expected_elbo,
                        "pfn_conf_elbo": pfn_conf_elbo,
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
                                            bounds_func(phi_prior_dict), n_kl_samples=None,
                                            legend=["Exact Prior", "Distribution Transformer"])

            if len(competitor_kwargs) == 0:
                model_posterior_plot = plot_distributions(model_posterior, None, model.sample_space_transform,
                                                          n_kl_samples=None)

            if "vi" in competitor_kwargs:
                vi = VI(model.state_size, prior, complete_distribution.observation_model, inverse_transform)
                torch.set_grad_enabled(True)
                start_time = time()
                for i in range(repeats):
                    vi.fit(z, epoch=i, **competitor_kwargs["vi"])
                vi_single_time = time() - start_time
                torch.set_grad_enabled(False)
                vi_posterior = vi.distribution()
                vi_posterior_plot = plot_distributions(vi_posterior, model_posterior, model.sample_space_transform,
                                                       model.sample_space_transform, bounds_func(phi_prior_dict),
                                                       n_kl_samples=None, legend=["SVI", "Distribution Transformer"])

            if "pfns" in competitor_kwargs:
                pfn = pfn.cpu()
                start_time = time()
                phi_out = pfn(**z)
                pfn_single_inference_time = time() - start_time
                pfn_posterior = RiemannDistribution(phi_out, pfn.borders, pfn.infinite_support)
                pfn_posterior_plot = plot_distributions(pfn_posterior, model_posterior, None,
                                                        model.sample_space_transform, bounds_func(phi_prior_dict),
                                                        n_kl_samples=None, legend=["PFN", "Distribution Transformer"])

            if _run is not None:
                prior_plot.savefig(_run.observers[0].dir + f"\\prior_plot_{n_components}.pdf", format="pdf")
                if len(competitor_kwargs) == 0:
                    model_posterior_plot.savefig(_run.observers[0].dir + f"\\model_posterior_plot_{n_components}.pdf", format="pdf")
                if "vi" in competitor_kwargs:
                    _run.info.update({
                        "vi_single_time": vi_single_time
                    })
                    vi_posterior_plot.savefig(_run.observers[0].dir + f"\\vi_posterior_plot_{n_components}.pdf", format="pdf")

                if "pfns" in competitor_kwargs:
                    pfn_posterior_plot.savefig(_run.observers[0].dir + f"\\pfn_posterior_plot_{n_components}.pdf", format="pdf")
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
        filtered_series_dict, _ = filter.filter(observation_series, motion_model.x0_distribution)
        model_density = GaussianMixtureModel(**filtered_series_dict)
        model_inference_time = (time() - start_time) / series_length

        model_nll = -model_density.log_prob(series)
        model_expected_nll = model_nll.mean().item()
        model_conf_nll = model_nll.std().item() * 1.96 / sqrt(n_test_series * series_length)

        if "ekf" in competitor_kwargs:
            ekf = EKF(model.state_size, motion_model, **observation_model)

            start_time = time()
            ekf_filtered_series_dict, _ = ekf.filter(observation_series, motion_model.x0_distribution)
            ekf_density = MultivariateNormal(**ekf_filtered_series_dict)
            ekf_inference_time = (time() - start_time) / series_length

            ekf_nll = -ekf_density.log_prob(series)
            ekf_expected_nll = ekf_nll.mean().item()
            ekf_conf_nll = ekf_nll.std().item() * 1.96 / sqrt(n_test_series * series_length)

        if "particle_filter" in competitor_kwargs:
            particle_filter = ParticleFilter(model.state_size, motion_model, **observation_model,
                                             **competitor_kwargs["particle_filter"])

            start_time = time()
            particles, _ = particle_filter.filter(observation_series, motion_model.x0_distribution)
            particle_filter_distribution = particle_filter.fit_density(particles)
            particle_filter_inference_time = (time() - start_time) / series_length


            particle_filter_nll = -particle_filter_distribution.log_prob(series)
            particle_filter_expected_nll = particle_filter_nll.mean().item()
            particle_filter_conf_nll = particle_filter_nll.std().item() * 1.96 / sqrt(n_test_series * series_length)


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
        filtered_series_dict, _ = filter.filter(observation_series, motion_model.x0_distribution)
        model_filter_distribution = GaussianMixtureModel(**filtered_series_dict)
        model_single_inference_time = (time() - start_time) / series_length

        if "ekf" in competitor_kwargs:
            start_time = time()
            ekf_filtered_series_dict, _ = ekf.filter(observation_series, motion_model.x0_distribution)
            ekf_filter_distribution = MultivariateNormal(**ekf_filtered_series_dict)
            ekf_single_inference_time = (time() - start_time) / series_length

        if "particle_filter" in competitor_kwargs:
            start_time = time()
            particles, _ = particle_filter.filter(observation_series, motion_model.x0_distribution)
            particle_filter_distribution = particle_filter.fit_density(particles)
            particle_filter_single_inference_time = (time() - start_time) / series_length

        if _run is not None:
            _run.info.update({
                "model_inference_time": model_inference_time,
                "model_single_inference_time": model_single_inference_time,
                "model_expected_nll": model_expected_nll,
                "model_conf_nll": model_conf_nll,
                "model_size": model_size,
            })
            if "ekf" in competitor_kwargs:
                _run.info.update({
                    "ekf_inference_time": ekf_inference_time,
                    "ekf_single_inference_time": ekf_single_inference_time,
                    "ekf_expected_nll": ekf_expected_nll,
                    "ekf_conf_nll": ekf_conf_nll,
                })
            if "particle_filter" in competitor_kwargs:
                _run.info.update({
                    "particle_filter_inference_time": particle_filter_inference_time,
                    "particle_filter_single_inference_time": particle_filter_single_inference_time,
                    "particle_filter_expected_nll": particle_filter_expected_nll,
                    "particle_filter_conf_nll": particle_filter_conf_nll,
                })

        # Plotting first dimension of state space
        if plotting_kwargs is not None:
            # Select dimension
            dim = plotting_kwargs["dim"]

            filtered_series_dict["loc"] = filtered_series_dict["loc"][..., dim].unsqueeze(-1)
            filtered_series_dict[scale_parametrisation] = \
                filtered_series_dict[scale_parametrisation].diagonal(dim1=-2, dim2=-1)[..., dim].unsqueeze(
                    -1).unsqueeze(-1)
            filtered_series = encode_gmm_sample(filtered_series_dict, scale_parametrisation)
            model_filter_distribution = GaussianMixtureModel(**filtered_series_dict)

            """
            bounds = list(zip(*[gmm_bounds_func(decode_gmm_sample(dist, scale_parametrisation))
                                for dist in filtered_series]))
            bounds = (max(bounds[0]), min(bounds[1]))
            bounds = (max(bounds[0], series.max().item() + 1), min(bounds[1], series.min().item() - 1))
            """

            bounds = None

            model_series_plot = plot_filtered_series(model_filter_distribution, series[..., dim].unsqueeze(-1), bounds,
                                                     **plotting_kwargs)

            if "ekf" in competitor_kwargs and "particle_filter" in competitor_kwargs:
                ekf_filtered_series_dict["loc"] = ekf_filtered_series_dict["loc"][..., dim].unsqueeze(-1)
                ekf_filtered_series_dict["covariance_matrix"] = \
                    ekf_filtered_series_dict["covariance_matrix"].diagonal(dim1=-2, dim2=-1)[..., dim].unsqueeze(
                        -1).unsqueeze(-1)
                ekf_filter_distribution = MultivariateNormal(**ekf_filtered_series_dict)
                ekf_series_plot = plot_filtered_series(ekf_filter_distribution, series[..., dim].unsqueeze(-1), bounds,
                                                       **plotting_kwargs)
                combined_series_plot = plot_filtered_series([ekf_filter_distribution,
                                                             model_filter_distribution],
                                                            series[..., dim].unsqueeze(-1), bounds,
                                                            cmaps=["OrRd", "BuPu"],
                                                            legend_labels=["EKF Filter Density",
                                                                           "Distribution Transformer Filter Density"],
                                                            **plotting_kwargs)

            if _run is not None:
                model_series_plot.savefig(_run.observers[0].dir + "\\model_series_plot.pdf", format="pdf")

                if "ekf" in competitor_kwargs:
                    ekf_series_plot.savefig(_run.observers[0].dir + "\\ekf_series_plot.pdf", format="pdf")
                    combined_series_plot.savefig(_run.observers[0].dir + "\\combined_series_plot.pdf", format="pdf")
