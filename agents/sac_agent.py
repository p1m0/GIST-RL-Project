from typing import Callable, Optional, Sequence, Tuple
import copy

import torch
from torch import nn
import numpy as np

import infrastructure.pytorch_util as ptu

from networks.sac_mlp_policy import MLPPolicy
from networks.sac_state_action_value_critic import StateActionCritic


class SoftActorCritic(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        discount: float,
        n_layers: int = 2,
        hidden_size: int = 64,
        soft_target_update_rate: Optional[float] = None,
        # Actor-critic configuration
        num_actor_samples: int = 1,
        num_critic_updates: int = 1,
        # Settings for multiple critics
        num_critic_networks: int = 2,
        # Soft actor-critic
        use_entropy_bonus: bool = True,
        temperature: float = 0.05,
        backup_entropy: bool = True,
    ):
        super().__init__()

        self.actor = MLPPolicy(
            ac_dim=action_dim,
            ob_dim=np.prod(observation_shape),
            n_layers=n_layers,
            layer_size=hidden_size,
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.actor_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.actor_optimizer, factor=1.0)

        self.critics = nn.ModuleList(
            [
                StateActionCritic(
                    ob_dim=np.prod(observation_shape),
                    ac_dim=action_dim,
                    n_layers=n_layers,
                    size=hidden_size,
                )
                for _ in range(num_critic_networks)
            ]
        )

        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=3e-4)
        self.critic_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.critic_optimizer, factor=1.0)
        self.target_critics = nn.ModuleList(
            [
                StateActionCritic(
                    ob_dim=np.prod(observation_shape),
                    ac_dim=action_dim,
                    n_layers=n_layers,
                    size=hidden_size,
                )
                for _ in range(num_critic_networks)
            ]
        )
        self.update_target_critic()

        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.discount = discount
        self.num_critic_networks = num_critic_networks
        self.use_entropy_bonus = use_entropy_bonus
        self.temperature = temperature
        self.num_actor_samples = num_actor_samples
        self.num_critic_updates = num_critic_updates
        self.soft_target_update_rate = soft_target_update_rate
        self.backup_entropy = backup_entropy

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the action for a given observation.
        """
        with torch.no_grad():
            observation = ptu.from_numpy(observation)[None]

            action_distribution: torch.distributions.Distribution = self.actor(observation)
            action: torch.Tensor = action_distribution.sample()

            assert action.shape == (1, self.action_dim), action.shape
            return ptu.to_numpy(action).squeeze(0)

    def critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) Q-values for the given state-action pair.
        """
        return torch.stack([critic(obs, action) for critic in self.critics], dim=0)

    def target_critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) target Q-values for the given state-action pair.
        """
        return torch.stack(
            [critic(obs, action) for critic in self.target_critics], dim=0
        )

    def q_backup_strategy(self, next_qs: torch.Tensor) -> torch.Tensor:
        """
        Handle Q-values from multiple different target critic networks to produce target values.

        Parameters:
            next_qs (torch.Tensor): Q-values of shape (num_critics, batch_size). 
                Leading dimension corresponds to target values FROM the different critics.
        Returns:
            torch.Tensor: Target values of shape (num_critics, batch_size). 
                Leading dimension corresponds to target values FOR the different critics.
        """

        assert (
            next_qs.ndim == 2
        ), f"next_qs should have shape (num_critics, batch_size) but got {next_qs.shape}"
        num_critic_networks, batch_size = next_qs.shape
        assert num_critic_networks == self.num_critic_networks

        next_qs = torch.roll(next_qs, shifts=1, dims=0)

        # If our backup strategy removed a dimension, add it back in explicitly
        # (assume the target for each critic will be the same)
        if next_qs.shape == (batch_size,):
            next_qs = next_qs[None].expand((self.num_critic_networks, batch_size)).contiguous()

        assert next_qs.shape == (
            self.num_critic_networks,
            batch_size,
        ), next_qs.shape
        return next_qs

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        Update the critic networks by computing target values and minimizing Bellman error.
        """
        (batch_size,) = reward.shape

        with torch.no_grad():
            next_action_distribution: torch.distributions.Distribution = self.actor(next_obs)
            next_action = next_action_distribution.sample()
            next_qs = self.target_critic(next_obs, next_action)

            # Handle Q-values from multiple different target critic networks (if necessary)
            # For double-Q
            next_qs = self.q_backup_strategy(next_qs)

            assert next_qs.shape == (
                self.num_critic_networks,
                batch_size,
            ), next_qs.shape

            next_action_entropy = self.entropy(next_action_distribution)
            if self.use_entropy_bonus and self.backup_entropy:
                next_qs += self.temperature * next_action_entropy

            target_values: torch.Tensor = reward + (1 - done.float()) * self.discount * next_qs
            assert target_values.shape == (
                self.num_critic_networks,
                batch_size
            )

        # Predict Q-values
        q_values = self.critic(obs, action)
        assert q_values.shape == (self.num_critic_networks, batch_size), q_values.shape

        # Compute loss
        loss: torch.Tensor = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "entropy_critic": next_action_entropy.mean().item() if self.use_entropy_bonus else 0.0,
        }

    def entropy(self, action_distribution: torch.distributions.Distribution):
        """
        Compute the (approximate) entropy of the action distribution for each batch element.
        """
        actions = action_distribution.rsample((10,))
        log_probs = action_distribution.log_prob(actions)
        
        return -log_probs.mean(dim=0)

    def actor_loss_reparametrize(self, obs: torch.Tensor):
        action_distribution: torch.distributions.Distribution = self.actor(obs)

        action = action_distribution.rsample()
        q_values = self.critic(obs, action)
        q_values_mean = torch.mean(q_values, axis=0)
        loss = -torch.mean(q_values_mean)

        return loss, torch.mean(self.entropy(action_distribution))

    def update_actor(self, obs: torch.Tensor):
        """
        Update the actor by one gradient step using either REPARAMETRIZE or REINFORCE.
        """
        loss, entropy = self.actor_loss_reparametrize(obs)

        # Add entropy if necessary
        loss -= self.temperature * entropy

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": loss.item(), "entropy": entropy.item()}

    def update_target_critic(self):
        self.soft_update_target_critic(1.0)

    def soft_update_target_critic(self, tau):
        for target_critic, critic in zip(self.target_critics, self.critics):
            for target_param, param in zip(
                target_critic.parameters(), critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        """
        Update the actor and critic networks.
        """
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        rewards = ptu.from_numpy(rewards)
        next_observations = ptu.from_numpy(next_observations)
        dones = ptu.from_numpy(dones)

        critic_infos = []
        for _ in range(self.num_critic_updates):
            critic_info = self.update_critic(
                observations, actions, rewards, next_observations, dones
            )
            critic_infos.append(critic_info)
        actor_info = self.update_actor(observations)

        self.soft_update_target_critic(self.soft_target_update_rate)

        # Average the critic info over all of the steps
        critic_info = {
            k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0]
        }

        # Deal with LR scheduling
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

        return {
            **actor_info,
            **critic_info,
            "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            "critic_lr": self.critic_lr_scheduler.get_last_lr()[0],
        }
