from typing import Optional, Sequence
import numpy as np
import torch

from networks.pg_policy import MLPPolicyPG
from networks.pg_critic import ValueCritic
from infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)
        obs = np.concatenate(obs, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        terminals = np.concatenate(terminals, axis=0)
        q_values = np.concatenate(q_values, axis=0)
        
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )
        info: dict = self.actor.update(obs, actions, advantages)
        if self.critic is not None:
            critic_loss = 0
            for _ in range(self.baseline_gradient_steps):
                critic_info: dict = self.critic.update(obs, q_values)
                critic_loss += critic_info['Baseline Loss']
            
            critic_loss /= self.baseline_gradient_steps
            info.update({'Baseline Loss': critic_loss})
        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            q_values = [self._discounted_return(reward) for reward in rewards]
        else:
            q_values = [self._discounted_reward_to_go(reward) for reward in rewards]

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            advantages = q_values
        else:
            values = ptu.to_numpy(self.critic(ptu.from_numpy(obs)))
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                advantages = q_values - values
            else:
                batch_size = obs.shape[0]
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    delta = rewards[i] + self.gamma * values[i+1] * (1 - terminals[i]) - values[i]
                    if terminals[i]:
                        advantages[i] = delta
                    else:
                        advantages[i] = delta + self.gamma * self.gae_lambda * advantages[i+1]

                advantages = advantages[:-1]

        if self.normalize_advantages:
            mean = np.mean(advantages)
            std = np.std(advantages)
            advantages = advantages - mean
            if (std != 0):
                advantages = advantages / std

        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        sum_value = 0
        for t in range(len(rewards)):
            sum_value += rewards[t] * (self.gamma ** t)
        return [sum_value] * len(rewards)

    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        discounted_rewards = []
        for t in range(len(rewards)):
            sum_value = 0
            for t_ in range(t, len(rewards)):
                sum_value += rewards[t_] * (self.gamma ** (t_ - t))
            discounted_rewards.append(sum_value)

        return discounted_rewards
