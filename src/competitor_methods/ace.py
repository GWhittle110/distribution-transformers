import copy
from competitor_methods.ace_utils.utils import AttrDict
import torch
from .ace_utils import ACEBaseTransformer, EmbedderMarker, TNPDEncoder, MixtureGaussian
from distributions.distributions import CompleteDistribution, MixtureSameFamily, MultivariateNormal, Normal

def get_ace_model(
    d_model,
    dim_feedforward,
    n_head,
    dropout,
    num_layers,
    dim_y,
    dim_yc,
    dim_xc,
    num_components,
    emb_depth,
    num_latent,
    activation,
    separate_embeddings=False,
):
     
    encoder = TNPDEncoder(d_model, dim_feedforward, n_head=n_head, dropout=dropout, num_layers=num_layers, activation=activation)
    head = MixtureGaussian(dim_y, d_model, dim_feedforward, num_components)
    embedder = EmbedderMarker(dim_xc, dim_yc, num_latent, d_model, d_model, emb_depth, separate_embeddings=separate_embeddings)
    model = ACEBaseTransformer(embedder, encoder, head)
    
    return model


def move_query_to_target(x, z): 
    z = copy.deepcopy(z)
    assert 'query' in z, "No query in z"

    query = z.pop('query')
    x[:,0,:query.shape[-1]] = query.reshape((x.shape[0], x.shape[2]-1))  # move query to x
    return x, z
    
    
def pad(x, total_input_features):
    x_padded = torch.cat([torch.zeros(x.shape[:-1] + (total_input_features - x.shape[-1],)).to(x), x], dim=-1)  # [..., feature_dim]
    return x_padded
    

def convert_distribution_sample_to_ace_input(phi, x, z, batch_size, xc, yc, predict=False, randomise_target=False):
    batch = AttrDict()
   
    phi_num_latent = phi.shape[1]
    total_input_features = xc + yc - 1  # -1 because xc includes index variable
    device = phi.device
    
    x_padded = pad(x.reshape(batch_size, -1, yc), total_input_features)
    
    if 'query' in z:
        x_padded, z = move_query_to_target(x_padded, z)
        query_in_x = True
    else:
        query_in_x = False

    z_keys = list(sorted(z.keys()))
    z_padded = [
        pad(z[key], total_input_features).reshape((batch_size, -1, total_input_features))  # [batch_size, z_num_latent, feature_dim]
        for key in z_keys
    ]
    z_stacked = torch.cat(z_padded, dim=1)  # [batch_size, z_num_latent, 1]
    z_stacked = z_stacked.reshape(batch_size, -1, total_input_features)
    z_num_latent = z_stacked.shape[1]
    
    phi_padded = pad(phi.reshape(batch_size, -1, yc), total_input_features)  # [batch_size, phi_num_latent, feature_dim]

    all_latents = torch.cat([phi_padded, x_padded, z_stacked], dim=1)  # [batch_size, num_latent, 1]
    index_variable = torch.arange(2, 2 + all_latents.shape[1]).unsqueeze(0).expand(batch_size, -1).reshape(batch_size, -1, 1).to(device=device)  # [batch_size, num_latent, 1]

    if "dataset" in z_keys:
        data_pos_within_z = z_keys.index("dataset")
        data_start = phi.shape[1] + x.shape[1] + sum(z[key].shape[1] for key in z_keys[:data_pos_within_z])
        index_variable = index_variable.clone()
        index_variable[:, data_start:data_start + z["dataset"].shape[1], :] = torch.ones((batch_size, z["dataset"].shape[1], 1), device=device)  # mark data with 1

    if query_in_x:
        query_start = phi.shape[1]
        index_variable = index_variable.clone()
        index_variable[:, query_start:query_start + 1, :] = torch.ones((batch_size, 1, 1), device=device)  # mark query with 1

    xyl = torch.cat([index_variable, all_latents], dim=-1)  # [batch_size, num_latent, feature_dim+1]

    if predict or not(randomise_target):
        # drop x from the input
        xyc = torch.cat([xyl[:, :phi_num_latent, :], xyl[:, -z_num_latent:, :]], dim=1)  # [batch_size, num_latent - x_num_latent, feature_dim+1]
        xyt = xyl[:, phi_num_latent:-z_num_latent, :]  # [batch_size, 1, feature_dim+1]
    else:
        random_permutation = torch.argsort(torch.rand(*xyl.shape), dim=1)
        xyl = torch.gather(xyl, 1, random_permutation)  # [batch_size, num_latent, feature_dim+1]
    
        num_latent = xyl.shape[1]
        num_ctx = num_latent - 1

        xyc = xyl[:, :num_ctx, :]  # [batch_size, num_ctx, feature_dim+1]
        xyt = xyl[:, num_ctx:, :]  # [batch_size, num_targets, feature_dim+1]

    # Separate features and labels for context and target points
    batch.xc = xyc[:, :, :-1]  # Context features: [batch_size, num_ctx, feature_dim]
    batch.yc = xyc[:, :, -1:]  # Context labels: [batch_size, num_ctx, 1]

    batch.xt = xyt[:, :, :-1]  # Target features: [batch_size, num_targets, feature_dim]
    batch.yt = xyt[:, :, -1:]  # Target labels: [batch_size, num_targets, 1]
    
    return batch


def predict_w_ace(phi, x, z, model):
    batch_size = x.shape[0] if len(x.shape) > 0 else 1
    batch = convert_distribution_sample_to_ace_input(phi, x, z, batch_size, predict=True, xc=model.embedder.dim_xc, yc=model.embedder.dim_yc)  # [batch_size, num_latent - x_num_latent, feature_dim+1]

    with torch.no_grad():
        out = model(batch, predict=True)  # [batch_size, num_targets, dim_y], [batch_size, num_targets, dim_y]

    if out["mixture_means"].shape[1] == 1:
        means = out['mixture_means'].squeeze(-2) # [batch_size, num_comp]
        stds = out['mixture_stds'].squeeze(-2)
        weights = out['mixture_weights'].squeeze(-2)

        var = stds ** 2
        means = means.unsqueeze(-1)
        var = var.unsqueeze(-1).unsqueeze(-1)

        component_distribution = MultivariateNormal(means, var)
    else:
        means = out['mixture_means'] # [batch_size, num_comp, dim_y]
        stds = out['mixture_stds']
        weights = out['mixture_weights']

        var = stds ** 2

        component_distribution = Normal(means, var)
    
    mixture_distribution = torch.distributions.Categorical(probs=weights)
    complete_distribution = MixtureSameFamily(mixture_distribution, component_distribution)

    return complete_distribution


def sampler_latent_only(
    complete_distribution,
    batch_size,
    xc,
    yc,
    randomise_target=False,
    sample_space_transform=None,
):

    # Retrieve data
    phi, x, z = complete_distribution.sample(
        torch.Size([batch_size])
    )
    if sample_space_transform is not None:
        x = sample_space_transform(x)

    batch = convert_distribution_sample_to_ace_input(phi, x, z, batch_size, xc, yc, randomise_target=randomise_target)  # [batch_size, num_latent, feature_dim+1]

    return batch


class Sampler(object):
    def __init__(
        self,
        complete_distribution,
        batch_size,
        xc,
        yc,
        randomise_target=False,
        ctx_tar_sampler="one_side",
        sample_space_transform=None,
        **kwargs,
    ):
        self.complete_distribution = complete_distribution
        self.batch_size = batch_size
        self.randomise_target = randomise_target
        self.ctx_tar_sampler = ctx_tar_sampler
        self.sample_space_transform = sample_space_transform
        self.xc = xc
        self.yc = yc
        self.kwargs = kwargs

    def sample(self):

        # Call the sampler function with the necessary parameters
        batch = sampler_latent_only(
            complete_distribution=self.complete_distribution,
            batch_size=self.batch_size,
            xc=self.xc,
            yc=self.yc,
            randomise_target=self.randomise_target,
            sample_space_transform=self.sample_space_transform,
        )
        return batch


def convert_complete_distribution_to_ace_sampler(complete_distribution: CompleteDistribution, xc, yc, batch_size: int = 16, randomise_target=False, sample_space_transform=None) -> Sampler:

    return Sampler(complete_distribution=complete_distribution, batch_size=batch_size, xc=xc, yc=yc, randomise_target=randomise_target, sample_space_transform=sample_space_transform)

