"""
Experiment evaluating method on problem of finding posterior for GP hyperparameters
"""
import torch
from copy import copy

from competitor_methods.pfns import PFN
from workflows.train import train_pfn


from model.embeddings import  ObservationEmbedding
from experiments.sources.static.gp_predictive_hyperprior import (MeanScaleMetaPrior,
                                                                 CompleteDistributionGPPredictive,
                                                                 GPPredictiveObservationModel)


def run(n_components: int,
        state_size: int,
        meta_prior_kwargs: dict,
        component_embedding_kwargs: dict,
        observation_embedding_kwargs: dict[str, dict],
        distribution_embedding_kwargs: dict,
        transformer_kwargs: dict,
        training_kwargs: dict,
        testing_kwargs: dict,
        _run=None,
        *args, **kwargs):
    
    marginalise_y = kwargs.get("marginalise_y", False)
    marginalise_lengthscale = kwargs.get("marginalise_lengthscale", False)
    
    assert not (marginalise_y and marginalise_lengthscale)

    observation_model = {observation_type: GPPredictiveObservationModel(observation_type=observation_type)
                         for observation_type in ["dataset", "query"]}


    # Distribution transformer
    d_model = transformer_kwargs["d_model"]

    meta_prior_pfn = MeanScaleMetaPrior(marginalise_lengthscale=marginalise_lengthscale, marginalise_y=marginalise_y, **meta_prior_kwargs)
    complete_distribution_pfn = CompleteDistributionGPPredictive(meta_prior_pfn,  marginalise_y=marginalise_y, marginalise_lengthscale=marginalise_lengthscale, **observation_model)
    competitor_kwargs = testing_kwargs["competitor_kwargs"]
    
    if "pfns"in competitor_kwargs:
        pfn_kwargs = copy(competitor_kwargs["pfns"])
        del pfn_kwargs["training_kwargs"]
        pfn_observation_embedding = {key: ObservationEmbedding(d_model=d_model, observation_size= (meta_prior_kwargs["x_dimensions"] + (1 if key=="dataset" else 0)), **kwargs)
                             for key, kwargs in observation_embedding_kwargs.items()}
        pfn = PFN(**pfn_kwargs, **pfn_observation_embedding)

        if kwargs.get("resume_path", False):
            pfn.load_state_dict(torch.load(kwargs.get("resume_path"), weights_only=True))

        torch.set_grad_enabled(True)
        pfn, _ = train_pfn(pfn, complete_distribution_pfn, _run=_run, **competitor_kwargs["pfns"]["training_kwargs"])
        torch.set_grad_enabled(False)

    #test_gp(model, complete_distribution,
    #     bounds_func=partial(gmm_bounds_func,
    #                         scale_parametrisation=component_embedding_kwargs["scale_parametrisation"]),
    #     linspace_size=1000,
    #     hyperpior=True,
    #     _run=_run, **testing_kwargs)