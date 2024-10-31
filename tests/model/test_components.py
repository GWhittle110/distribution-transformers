import pytest
import torch

from model.components import Cholesky, PositiveDefinite


class TestCholesky:

    @pytest.mark.parametrize(
        ("state_size", "x", "jitter", "expected_output"),
        [
            (
                2,
                torch.tensor([1, 2, 3, 4], dtype=torch.float32),
                0.,
                torch.tensor([1, 0, 3, 4], dtype=torch.float32)
            ),
            (
                2,
                torch.tensor([1, 2, -3], dtype=torch.float32).broadcast_to(5, 3),
                0.1,
                torch.tensor([1.1, 0, 2, 3.1], dtype=torch.float32).broadcast_to(5, 4),
            )
        ]
    )
    def test_forward_from_flat(self, state_size: int, x: torch.Tensor, jitter: float, expected_output: torch.Tensor):
        component = Cholesky(state_size=state_size, jitter=jitter)
        chol = component(x)
        assert torch.equal(chol, expected_output)
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
                    torch.eye(2, dtype=torch.float32)
            )
        ]
    )
    def test_forward_from_matrix(self, state_size: int, x: torch.Tensor, jitter: float, expected_output: torch.Tensor):
        component = Cholesky(state_size=state_size, jitter=jitter)
        chol_mat = component(x)
        assert torch.equal(chol_mat, expected_output)
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
                torch.tensor([1, 3, 3, 25], dtype=torch.float32)
            ),
            (
                2,
                torch.tensor([1, 2, -3], dtype=torch.float32).broadcast_to(5, 3),
                0.1,
                torch.tensor([1.21, 2.2, 2.2, 13.61], dtype=torch.float32).broadcast_to(5, 4),
            )
        ]
    )
    def test_forward_from_flat(self, state_size: int, x: torch.Tensor, jitter: float, expected_output: torch.Tensor):
        component = PositiveDefinite(state_size=state_size, jitter=jitter)
        flat_mat = component(x)
        assert torch.equal(flat_mat, expected_output)
        mat = flat_mat.reshape(flat_mat.shape[:-1] + (state_size, state_size))
        assert torch.greater_equal(torch.linalg.det(mat), jitter ** (2 * state_size)).all()

    @pytest.mark.parametrize(
        ("state_size", "x", "jitter", "expected_output"),
        [
            (
                    2,
                    torch.eye(2, dtype=torch.float32),
                    0.,
                    torch.eye(2, dtype=torch.float32)
            ),
            (
                2,
                torch.zeros((2, 2), dtype=torch.float32),
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
