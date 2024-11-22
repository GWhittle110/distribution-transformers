"""
Experiment to validate method against closed form posterior of GMM prior with linear Gaussian observations

Procedure:
Sample ~10 observation models, state sizes and number of components from some distributions. For each:
    1) Train the variational transformer
    2) Sample ~1000 priors and observations as in training and compute posterior using variational transformer and
        analytical solution.
Compute average KL divergence KL[q || p]
"""

import torch
from time import time

from distributions.distributions import (GaussianMixtureModelConjugateMetaPrior, LinearGaussianObservationModel,
                                         CompleteDistribution, GaussianMixtureModel)
from distributions.utils import gmm_with_linear_gaussian_observations_posterior, kl_divergence, plot_distributions
from model.transformers import GMMConditionalTransformerModel
from model.train import train


def run(n_components: int, state_size: int, n_test_priors: int, n_kl_samples: int,
        meta_prior_params: dict, observation_covariance_matrix: list[list[float]],
        observation_matrix: list[list[float]], transformer_params: dict, training_params: dict, _run=None,
        *args, **kwargs):
    """
    Run an experiment comparing distribution transformers to the closed form posterior of a GMM prior under linear
    Gaussian observations.

    Args:
        n_components: Number of GMM components.
        state_size: Dimensionality of GMM.
        n_test_priors: Number of sampled priors to compare model posterior to closed form posterior on.
        n_kl_samples: Number of samples with which to compute the KL divergence between posteriors.
        meta_prior_params: Dictionary of parameters for the meta prior.
        observation_covariance_matrix: Observation covariance matrix, specified in list of lists format.
        observation_matrix: Observation matrix, specified in list of lists format.
        transformer_params: Dictionary of parameters for the transformer model.
        training_params: Dictionary of parameters for the training routine.
        _run: Sacred run object.

    """

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Meta-prior
    meta_prior = GaussianMixtureModelConjugateMetaPrior(state_size=state_size, n_components=n_components,
                                                        **meta_prior_params)

    # Observation model
    covariance_matrix = torch.tensor(observation_covariance_matrix, dtype=torch.float32)
    observation_matrix = torch.tensor(observation_matrix, dtype=torch.float32)
    observation_model = LinearGaussianObservationModel(observation_matrix=observation_matrix,
                                                       covariance_matrix=covariance_matrix)

    # Complete distribution
    complete_distribution = CompleteDistribution(meta_prior, observation_model)

    # Variational transformer
    model = GMMConditionalTransformerModel(n_components=n_components, state_size=state_size,
                                           n_observations=observation_model.n_observations, **transformer_params)

    before_training = time()
    model, last_epoch_metrics = train(model, complete_distribution, **training_params)
    training_time = time() - before_training
    model.to(device)

    with torch.no_grad():
        # Test inputs
        test_samples = complete_distribution.sample((n_test_priors,)).to(device)
        test_prior_params = complete_distribution.decode_sample(test_samples)["phi"]
        test_observations = complete_distribution.decode_sample(test_samples)["z"]
        test_priors = GaussianMixtureModel(**meta_prior.decode_sample(test_prior_params))

        # Exact solution
        observation_model.observation_matrix = observation_model.observation_matrix.to(device)
        observation_model.covariance_matrix = observation_model.covariance_matrix.to(device)
        exact_posterior = gmm_with_linear_gaussian_observations_posterior(test_priors, observation_model,
                                                                          test_observations)
        # Inference solution
        start_time = time()
        model_posterior_params = model(test_prior_params, test_observations)
        model_posterior = GaussianMixtureModel(**meta_prior.decode_sample(model_posterior_params))
        inference_time = time() - start_time

        kl_divergences = kl_divergence(model_posterior, exact_posterior, n_samples=n_kl_samples)
        prior_kl_divergences = kl_divergence(test_priors, exact_posterior, n_samples=n_kl_samples)
        print(f"Model mean KL divergence: {kl_divergences.mean().item()} \n"
              f"Prior mean KL divergence: {prior_kl_divergences.mean().item()}")

        _run.info.update({"training_epoch_metrics": last_epoch_metrics,
                          "training_time": training_time,
                          "test_inference_time": inference_time,
                          "model_mean_kl_divergence": kl_divergences.mean().item(),
                          "prior_mean_kl_divergence": prior_kl_divergences.mean().item(),
                          "model_std_kl_divergence": kl_divergences.std().item(),
                          "prior_std_kl_divergence": prior_kl_divergences.std().item()})

        if state_size == 1:
            plot_prior = GaussianMixtureModel(**meta_prior.decode_sample(test_prior_params[0].cpu()))
            prior_plot = plot_distributions(plot_prior, bounds=(-10, 10))
            prior_plot.savefig(_run.observers[0].dir + "\\prior_plot.png")

            plot_exact_posterior = gmm_with_linear_gaussian_observations_posterior(plot_prior, observation_model,
                                                                                   test_observations[0].cpu())
            plot_posterior = GaussianMixtureModel(**meta_prior.decode_sample(model_posterior_params[0].cpu()))
            posterior_plot = plot_distributions(plot_exact_posterior, plot_posterior, bounds=(-10, 10))
            posterior_plot.savefig(_run.observers[0].dir + "\\posterior_plot.png")
