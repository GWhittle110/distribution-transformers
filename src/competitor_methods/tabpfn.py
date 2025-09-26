import torch
from tabpfn import TabPFNRegressor  


def construct_tabpfn_inputs(
    phi, x, z
):
    batch_size = x.shape[0]

    phi_keys_sorted = sorted([key for key in phi.keys()])
    obs_keys_sorted = sorted([key for key in z.keys()])

    phi_stacked = torch.cat([phi[key].reshape(batch_size, -1) for key in phi_keys_sorted], dim=-1)
    z_stacked = torch.cat([z[key].reshape(batch_size, -1) for key in obs_keys_sorted], dim=-1)
    x = x.reshape(batch_size, -1)

    inp = torch.cat([phi_stacked, z_stacked], dim=-1)
    output = x

    return inp, output

def test_tabpfn(
    phi_dict_test, x_test, z_test,
    complete_distribution,
    tabpfn_trainsize
    ):
        
    phi_train, x_train, z_train = complete_distribution.sample((tabpfn_trainsize,))
    phi_prior_dict = complete_distribution.meta_prior.decode_sample(phi_train)

    X_train, Y_train = construct_tabpfn_inputs(phi_prior_dict, x_train, z_train)
    X_test, Y_test = construct_tabpfn_inputs(phi_dict_test, x_test, z_test)

    regressor = TabPFNRegressor()

    regressor.fit(X_train, Y_train)
    
    regressor_prediction = regressor.predict(X_test, output_type="full")
    
    output_distribution = TabPFNOutputDistribution(regressor_prediction)
    nll = -output_distribution.log_prob(Y_test)

    return output_distribution, nll, regressor


class TabPFNOutputDistribution(torch.distributions.Distribution):
    arg_constraints = {}
    support = torch.distributions.constraints.real
    has_rsample = False

    def __init__(self, regressor_prediction: dict):
        """
        Custom distribution class for TabPFN outputs.

        """
        batch_shape = regressor_prediction['logits'].shape[:-1]
        if len(batch_shape) == 1:
            batch_shape = torch.Size(())
        super().__init__(batch_shape=batch_shape, event_shape=torch.Size(()))
        self.regressor_prediction = regressor_prediction
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of value under the TabPFN output distribution.

        Args:
            value: Value at which to evaluate log probability of shape (...,).

        Returns:
            Log probability of shape (...,).
        """
        criterion = self.regressor_prediction['criterion'].cpu()
        logits = self.regressor_prediction['logits'].cpu()
        batch_size = value.shape[0]

        value = value.reshape(batch_size, -1)
        logits = logits.expand(batch_size, -1)
        return -criterion.forward(logits, value.cpu()).reshape(batch_size)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Sample from the TabPFN output distribution.

        Args:
            sample_shape: Shape of samples to draw.

        Returns:
            Samples from the TabPFN output distribution.
        """
        criterion = self.regressor_prediction['criterion'].cpu()
        logits = self.regressor_prediction['logits'].cpu()
        return criterion.sample(logits, sample_shape)
