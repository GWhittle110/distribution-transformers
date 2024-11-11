import pytest
import torch
from typing import Sequence

from model.distribution_embeddings import GaussianEmbedding


class TestGaussianEmbedding:

    @pytest.mark.parametrize(
        ("state_size", "d_model", "hidden_layer_sizes", "x"),
        [
            (
                2,
                10,
                None,
                torch.tensor([1, 0, 0, 1, 0, 0, 1], dtype=torch.float32),
            ),
            (
                2,
                10,
                [10, 20],
                torch.tensor([1, 0, 0, 1, 0, 0, 1], dtype=torch.float32).broadcast_to(5, 7),
            )
        ]
    )
    def test_forward(self, state_size: int, d_model: int, hidden_layer_sizes: Sequence, x: torch.Tensor):
        embedding_function = GaussianEmbedding(state_size=state_size, d_model=d_model,
                                               hidden_layer_sizes=hidden_layer_sizes)
        embedding = embedding_function(x)
        assert embedding.shape == x.shape[:-1] + (d_model,)

    @pytest.mark.parametrize(
        ("state_size", "d_model", "hidden_layer_sizes", "x", "jitter"),
        [
            (
                    2,
                    10,
                    None,
                    torch.ones(10, dtype=torch.float32),
                    0.,
            ),
            (
                    2,
                    10,
                    [10, 10],
                    torch.ones(5, 10),
                    0.1,
            )
        ]
    )
    def test_reverse(self, state_size: int, d_model: int, hidden_layer_sizes: Sequence,
                     x: torch.Tensor, jitter: float):
        batch_size = x.shape[:-1]
        embedding_function = GaussianEmbedding(state_size=state_size, d_model=d_model,
                                               hidden_layer_sizes=hidden_layer_sizes, jitter=jitter)
        decoded = embedding_function(x, reverse=True)
        print(decoded)
        scale = decoded[..., -state_size**2:].reshape(batch_size + (state_size, state_size))
        assert torch.greater_equal(torch.linalg.det(scale), jitter ** (2 * state_size)).all()
        assert torch.greater_equal(torch.diagonal(scale, dim1=-2, dim2=-1), 0).all()
