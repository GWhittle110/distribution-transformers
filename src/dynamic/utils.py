
import torch
from torch import Tensor
from torch.distributions import Distribution

import matplotlib.pyplot as plt
from typing import Optional


def plot_filtered_series(filter_distribution: Distribution,
                         series: Optional[Tensor] = None,
                         bounds: Optional[tuple[float, float]] = None,
                         n_vertical: int = 1000,
                         *args,
                         **kwargs) -> plt.Figure:
    plt.style.use(['seaborn-v0_8-paper', 'seaborn-v0_8-whitegrid'])

    if bounds is None:
        assert series is not None, "series_bounds cannot be inferred if true_series is not provided"
        bounds = (series.max().item() + 1., series.min().item() - 1.)

    x = torch.arange(series.shape[0], dtype=torch.float32)
    y = torch.linspace(bounds[0], bounds[1], n_vertical)
    X, Y = torch.meshgrid(x, y)
    Z = filter_distribution.log_prob(Y.T.unsqueeze(-1)).exp().T
    fig = plt.figure()
    plt.pcolormesh(X, Y, Z, cmap="BuPu")
    plt.plot(series, "k")
    plt.colorbar()
    plt.show()
    return fig
