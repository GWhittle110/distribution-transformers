"""
Motion models for filtering equations
"""

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal, Distribution
from torch.types import _size

from typing import Callable


class MotionModel:

    def __init__(self, state_size: int,
                 mu: Callable[[Tensor, int], Tensor],
                 sigma: Callable[[Tensor, int], Tensor],
                 x0_distribution: Distribution,
                 noise_distribution: Distribution):
        """
        Base class for SDE-based state space motion models of the form x_k+1 = mu(x_k, k) + sigma(x_k, k)n_k where n_k
        is multivariate normal distributed with zero mean and identity covariance matrix.

        Args:
            state_size: Size of state space.
            mu: Drift component of motion SDE. Must return (batched) vector of shape state_size.
            sigma: Diffusion component of motion SDE. Must return (batched) matrix of shape state_size X noise_size
            x0_distribution: Distribution for initial state.
            noise_distribution: Distribution for noise process noise.

        """
        self.state_size = state_size
        self.mu = mu
        self.sigma = sigma

        self.x0_distribution = x0_distribution
        self.noise_distribution = noise_distribution

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        """
        Sample trajectory. Trajectory length is given by first dimension of sample_shape.

        Args:
            sample_shape: Sample shape, in form (batch dims | trajectory length).

        Returns:
            Sampled trajectory.

        """
        assert sample_shape[0] > 1

        x0_sample = self.x0_distribution.sample(sample_shape[1:])
        noise_samples = self.noise_distribution.sample(sample_shape)

        trajectory_sample = torch.empty(*sample_shape, self.state_size)
        trajectory_sample[0] = x0_sample

        for i, noise_sample in enumerate(noise_samples[1:], start=1):
            trajectory_sample[i] = (self.mu(trajectory_sample[i-1], i-1)
                                    + torch.einsum("...ij, ...j -> ...i", self.sigma(trajectory_sample[i-1], i-1),
                                                   noise_sample))

        return trajectory_sample


class TimeInvariantMotionModel(MotionModel):

    def __init__(self, state_size: int,
                 mu: Callable[[Tensor], Tensor],
                 sigma: Callable[[Tensor], Tensor],
                 x0_distribution: Distribution,
                 noise_distribution: Distribution):
        """
        Base class for time-invariant SDE-based state space motion models of the form x_k+1 = mu(x_k) + sigma(x_k)n_k
        where n_k is distributed according to noise_distribution

        Args:
            state_size: Size of state space.
            mu: Drift component of motion SDE. Must return (batched) vector of shape state_size.
            sigma: Diffusion component of motion SDE. Must return (batched) matrix of shape state_size X noise_size,
                where noise_size is the event shape of noise_distribution.
            x0_distribution: Distribution for initial state.
            noise_distribution: Distribution for noise process noise.

        """
        super().__init__(state_size,
                         lambda x, k: mu(x),
                         lambda x, k: sigma(x),
                         x0_distribution,
                         noise_distribution)


class LTIMotionModel(TimeInvariantMotionModel):

    def __init__(self, state_transition_matrix: Tensor,
                 process_noise_scale_cholesky: Tensor,
                 x0_distribution: MultivariateNormal):
        """
        Linear time-invariant SDE-based state space motion models of the form x_k+1 = Ax_k + sqrt(Q)n_k
        where n_k is multivariate normal distributed with zero mean and identity covariance matrix.

        Args:
            state_transition_matrix: State transition matrix, A. Must be of shape state_size X state_size.
            process_noise_scale_cholesky: Cholesky decomposition of process noise covariance matrix, sqrt(Q). Must be of
                shape state_size X noise_size.
            x0_distribution: Distribution for initial state.

        """
        state_size = state_transition_matrix.shape[-1]
        noise_size = process_noise_scale_cholesky.shape[-1]

        super().__init__(state_size,
                         lambda x: torch.einsum("...ij, ...j -> ...i", state_transition_matrix, x),
                         lambda x: process_noise_scale_cholesky,
                         x0_distribution,
                         MultivariateNormal(torch.zeros(noise_size), torch.eye(noise_size)))

        self.state_transition_matrix = state_transition_matrix
        self.process_noise_covariance_matrix = process_noise_scale_cholesky @ process_noise_scale_cholesky.mT
        self.x0_distribution: MultivariateNormal = x0_distribution
