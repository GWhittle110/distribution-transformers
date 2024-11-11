import pytest
import torch
from torch import nn

from model.components import Cholesky, PositiveDefinite, Logit


class TestCholesky:

    @pytest.mark.parametrize(
        ("state_size", "x", "jitter", "expected_output"),
        [
            (
                2,
                torch.tensor([1, 2, 3, 4], dtype=torch.float32),
                0.,
                torch.tensor([2.7183,  0.0000,  3.0000, 54.5981], dtype=torch.float32)
            ),
            (
                2,
                torch.tensor([1, 2, -3], dtype=torch.float32).broadcast_to(5, 3),
                0.1,
                torch.tensor([2.8183, 0.0000, 2.0000, 0.1498], dtype=torch.float32).broadcast_to(5, 4),
            )
        ]
    )
    def test_forward_from_flat(self, state_size: int, x: torch.Tensor, jitter: float, expected_output: torch.Tensor):
        component = Cholesky(state_size=state_size, jitter=jitter)
        chol = component(x)
        assert torch.allclose(chol, expected_output, atol=1e-4)
        chol_mat = chol.reshape(chol.shape[:-1] + (state_size, state_size))
        assert torch.greater_equal(torch.linalg.det(chol_mat), jitter ** state_size).all()
        assert torch.allclose(torch.linalg.cholesky(torch.einsum("...ij,...kj->...ik", chol_mat, chol_mat)),
                              chol_mat)

    @pytest.mark.parametrize(
        ("state_size", "x", "jitter", "expected_output"),
        [
            (
                    2,
                    torch.eye(2, dtype=torch.float32),
                    0.,
                    torch.e * torch.eye(2, dtype=torch.float32)
            )
        ]
    )
    def test_forward_from_matrix(self, state_size: int, x: torch.Tensor, jitter: float, expected_output: torch.Tensor):
        component = Cholesky(state_size=state_size, jitter=jitter)
        chol_mat = component(x)
        assert torch.allclose(chol_mat, expected_output, atol=1e-4)
        assert torch.greater_equal(torch.linalg.det(chol_mat), jitter ** state_size).all()
        assert torch.equal(torch.linalg.cholesky(torch.einsum("...ij,...kj->...ik", chol_mat, chol_mat)),
                           chol_mat)


class TestPositiveDefinite:

    @pytest.mark.parametrize(
        ("state_size", "x", "jitter", "expected_output"),
        [
            (
                2,
                torch.tensor([1, 2, 3, 4], dtype=torch.float32),
                0.,
                torch.tensor([7.3891, 8.1548, 8.1548, 2989.9578], dtype=torch.float32)
            ),
            (
                2,
                torch.tensor([1, 2, -3], dtype=torch.float32).broadcast_to(5, 3),
                0.1,
                torch.tensor([7.9427, 5.6366, 5.6366, 4.0224], dtype=torch.float32).broadcast_to(5, 4),
            )
        ]
    )
    def test_forward_from_flat(self, state_size: int, x: torch.Tensor, jitter: float, expected_output: torch.Tensor):
        component = PositiveDefinite(state_size=state_size, jitter=jitter)
        flat_mat = component(x)
        assert torch.allclose(flat_mat, expected_output, atol=1e-4)
        mat = flat_mat.reshape(flat_mat.shape[:-1] + (state_size, state_size))
        assert torch.greater_equal(torch.linalg.det(mat), jitter ** (2 * state_size)).all()

    @pytest.mark.parametrize(
        ("state_size", "x", "jitter", "expected_output"),
        [
            (
                2,
                torch.eye(2, dtype=torch.float32),
                0.,
                torch.e ** 2 * torch.eye(2, dtype=torch.float32)
            ),
            (
                2,
                -1e6 * torch.eye(2, dtype=torch.float32),
                0.1,
                0.01 * torch.eye(2, dtype=torch.float32)
            )
        ]
    )
    def test_forward_from_matrix(self, state_size: int, x: torch.Tensor, jitter: float, expected_output: torch.Tensor):
        component = PositiveDefinite(state_size=state_size, jitter=jitter)
        mat = component(x)
        assert torch.allclose(mat, expected_output)
        assert torch.greater_equal(torch.linalg.det(mat), jitter ** (2 * state_size)).all()
        assert torch.equal(mat, mat.T)


class TestLogit:

    @pytest.mark.parametrize(
        ("x", "expected_result"),
        [
            (
                torch.tensor([0.1, 0.5, 0.9]),
                torch.tensor([-2.1972, 0, 2.1972])
            )
        ]
    )
    def test_forward(self, x: torch.Tensor, expected_result: torch.Tensor):
        logit = Logit()(x)
        assert torch.allclose(logit, expected_result, atol=1e-4)

    @pytest.mark.parametrize(
        "x",
        [
            torch.tensor([-1]),
            torch.tensor([2])
        ]
    )
    def test_domain(self, x: torch.Tensor):
        with pytest.raises(AssertionError):
            Logit()(x)
