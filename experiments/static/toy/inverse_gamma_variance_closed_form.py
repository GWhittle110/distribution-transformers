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
from torch.distributions import InverseGamma
from time import time

from distributions.distributions import (InverseGammaMetaPrior, MappedScaleGaussianObservationModel,
                                         GaussianMixtureModel, CompleteDistribution)
from distributions.special import ApproximateWarpedGMMMetaPrior, ApproximateCompleteDistribution
from distributions.utils import kl_divergence, plot_distributions
from model.transformers import GMMConditionalTransformerModel
from model.train import train
from model.distribution_converter import (DistributionConverter, TransformerLatentDistributionConverter,
                                          MLPLatentDistributionConverter, ResidualLatentDistributionConverter,
                                          MLPTransformerLatentDistributionConverter)
from model.distribution_embeddings import GaussianEmbedding, GammaEmbedding
from model.components import GMMReshaper


def run(n_components: int, n_test_priors: int, n_kl_samples: int,
        meta_prior_params: dict, observation_loc: list[float], transformer_params: dict, training_params: dict,
        use_distribution_converter: bool, _run=None, *args, **kwargs):
    """
    Run an experiment comparing distribution transformers to the closed form posterior of a GMM prior under linear
    Gaussian observations.

    Args:
        n_components: Number of GMM components.
        n_test_priors: Number of sampled priors to compare model posterior to closed form posterior on.
        n_kl_samples: Number of samples with which to compute the KL divergence between posteriors.
        meta_prior_params: Dictionary of parameters for the meta prior.
        observation_loc: Mean of observation distribution.
        transformer_params: Dictionary of parameters for the transformer model.
        training_params: Dictionary of parameters for the training routine.
        use_distribution_converter: Whether to use a distribution converter or an EM approximating GMM.
        _run: Sacred run object.

    """

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    observation_loc = torch.tensor(observation_loc, dtype=torch.float32)
    observation_model = MappedScaleGaussianObservationModel(observation_loc, scale_parametrisation="covariance_matrix",
                                                            mapping=None)
    if use_distribution_converter:
        meta_prior = InverseGammaMetaPrior(**meta_prior_params)
        complete_distribution = CompleteDistribution(meta_prior, observation_model)
        d_model = transformer_params["d_model"]
        distribution_embedding = GammaEmbedding(d_model, hidden_layer_sizes=[d_model])
        latent_converter = MLPLatentDistributionConverter(n_components=n_components, hidden_layer_sizes=[3 * d_model // 2], d_model=d_model)
        component_embedding = GaussianEmbedding(state_size=1,
                                                d_model=d_model,
                                                hidden_layer_sizes=[d_model // 2])
        mixture_reshaper = GMMReshaper(n_components, state_size=1, logit_weights=True)
        distribution_converter = DistributionConverter(distribution_embedding, latent_converter, component_embedding,
                                                       mixture_reshaper, torch.log)
        model = GMMConditionalTransformerModel(n_components=n_components, state_size=1,
                                               n_observations=observation_model.n_observations,
                                               distribution_converter=distribution_converter, **transformer_params)

    else:
        meta_prior = ApproximateWarpedGMMMetaPrior(InverseGammaMetaPrior(**meta_prior_params),
                                                   n_components=n_components,
                                                   transform=torch.log)


        complete_distribution = ApproximateCompleteDistribution(meta_prior, observation_model, sample_mode="exact")

        model = GMMConditionalTransformerModel(n_components=n_components, state_size=1,
                                               n_observations=observation_model.n_observations, **transformer_params)

    before_training = time()
    model, last_epoch_metrics = train(model, complete_distribution, **training_params)
    training_time = time() - before_training
    model.to(device)

    with torch.no_grad():
        if use_distribution_converter:
            # Test inputs
            test_samples = complete_distribution.sample((n_test_priors,)).to(device)
            test_exact_prior_params = complete_distribution.decode_sample(test_samples)["phi"]
            test_observations = complete_distribution.decode_sample(test_samples)["z"]
            test_prior_params = model.distribution_converter(test_exact_prior_params)
            test_priors = GaussianMixtureModel(**mixture_reshaper.decode_sample(test_prior_params))

            # Exact solution
            test_exact_priors = InverseGamma(
                **complete_distribution.meta_prior.decode_sample(test_exact_prior_params))
            exact_posterior_concentration = test_exact_priors.concentration + observation_model.n_observations / 2
            exact_posterior_rate = test_exact_priors.rate + 0.5 * ((test_observations -
                                                                    observation_model.loc.to(device)) ** 2).sum(dim=-1)
            exact_posterior = InverseGamma(concentration=exact_posterior_concentration, rate=exact_posterior_rate)

            # Inference solution
            start_time = time()
            model_posterior_params = model(test_prior_params, test_observations)
            model_posterior = GaussianMixtureModel(**mixture_reshaper.decode_sample(model_posterior_params))
            inference_time = time() - start_time

            kl_divergences = kl_divergence(exact_posterior, model_posterior, torch.log, n_kl_samples)
            prior_kl_divergences = kl_divergence(test_exact_priors, test_priors, torch.log, n_kl_samples)
            print(f"GMM approximation prior mean KL divergence: {prior_kl_divergences.mean().item()}\n"
                  f"Posterior mean KL divergence: {kl_divergences.mean().item()}\n")

            _run.info.update({"training_epoch_metrics": last_epoch_metrics,
                              "training_time": training_time,
                              "test_inference_time": inference_time,
                              "model_mean_kl_divergence": kl_divergences.mean().item(),
                              "prior_mean_kl_divergence": prior_kl_divergences.mean().item(),
                              "model_std_kl_divergence": kl_divergences.std().item(),
                              "prior_std_kl_divergence": prior_kl_divergences.std().item()})

            # For plotting
            max_x = lambda concentration, rate: (4 * rate / concentration + 1 / rate).item()
            plot_exact_prior = InverseGamma(
                **complete_distribution.meta_prior.decode_sample(test_exact_prior_params[0].cpu()))
            plot_prior = GaussianMixtureModel(**mixture_reshaper.decode_sample(test_prior_params[0].cpu()))
            prior_plot = plot_distributions(plot_exact_prior, plot_prior, torch.log,
                                            (0.001, max_x(*test_exact_prior_params[0].cpu())))
            prior_plot.savefig(_run.observers[0].dir + "\\prior_plot.png")

            plot_exact_posterior = InverseGamma(concentration=exact_posterior_concentration[0].cpu(),
                                                rate=exact_posterior_rate[0].cpu())
            plot_posterior = GaussianMixtureModel(**mixture_reshaper.decode_sample(model_posterior_params[0].cpu()))
            posterior_plot = plot_distributions(plot_exact_posterior, plot_posterior, torch.log,
                                                (0.001, max_x(exact_posterior_concentration[0].cpu(),
                                                              exact_posterior_rate[0].cpu())))
            posterior_plot.savefig(_run.observers[0].dir + "\\posterior_plot.png")

        else:
            # Test inputs
            test_samples = complete_distribution.sample((n_test_priors,)).to(device)
            test_prior_params = complete_distribution.decode_sample(test_samples)["phi"]
            test_observations = complete_distribution.decode_sample(test_samples)["z"]
            test_priors = GaussianMixtureModel(**meta_prior.decode_sample(test_prior_params))

            # Exact solution
            test_exact_prior_params = complete_distribution.meta_prior.exact_prior_samples.to(device)
            test_exact_priors = InverseGamma(
                **complete_distribution.meta_prior.meta_prior.decode_sample(test_exact_prior_params))
            exact_posterior_concentration = test_exact_priors.concentration + observation_model.n_observations / 2
            exact_posterior_rate = test_exact_priors.rate + 0.5 * ((test_observations -
                                                                    observation_model.loc.to(device)) ** 2).sum(dim=-1)
            exact_posterior = InverseGamma(concentration=exact_posterior_concentration, rate=exact_posterior_rate)

            # Inference solution
            start_time = time()
            model_posterior_params = model(test_prior_params, test_observations)
            model_posterior = GaussianMixtureModel(**meta_prior.decode_sample(model_posterior_params))
            inference_time = time() - start_time

            kl_divergences = kl_divergence(exact_posterior, model_posterior, torch.log, n_kl_samples)
            prior_kl_divergences = kl_divergence(test_exact_priors, test_priors, torch.log, n_kl_samples)
            print(f"GMM approximation prior mean KL divergence: {prior_kl_divergences.mean().item()}\n"
                  f"Posterior mean KL divergence: {kl_divergences.mean().item()}\n")

            _run.info.update({"training_epoch_metrics": last_epoch_metrics,
                              "training_time": training_time,
                              "test_inference_time": inference_time,
                              "model_mean_kl_divergence": kl_divergences.mean().item(),
                              "prior_mean_kl_divergence": prior_kl_divergences.mean().item(),
                              "model_std_kl_divergence": kl_divergences.std().item(),
                              "prior_std_kl_divergence": prior_kl_divergences.std().item()})

            # For plotting
            max_x = lambda concentration, rate: (4 * rate / concentration + 1 / rate).item()
            plot_exact_prior = InverseGamma(
                **complete_distribution.meta_prior.meta_prior.decode_sample(test_exact_prior_params[0].cpu()))
            plot_prior = GaussianMixtureModel(**meta_prior.decode_sample(test_prior_params[0].cpu()))
            prior_plot = plot_distributions(plot_exact_prior, plot_prior, torch.log,
                                            (0.001, max_x(*test_exact_prior_params[0].cpu())))
            prior_plot.savefig(_run.observers[0].dir+"\\prior_plot.png")

            plot_exact_posterior = InverseGamma(concentration=exact_posterior_concentration[0].cpu(),
                                                rate=exact_posterior_rate[0].cpu())
            plot_posterior = GaussianMixtureModel(**meta_prior.decode_sample(model_posterior_params[0].cpu()))
            posterior_plot = plot_distributions(plot_exact_posterior, plot_posterior, torch.log,
                                                (0.001, max_x(exact_posterior_concentration[0].cpu(),
                                                              exact_posterior_rate[0].cpu())))
            posterior_plot.savefig(_run.observers[0].dir + "\\posterior_plot.png")
