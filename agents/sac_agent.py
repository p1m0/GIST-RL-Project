from typing import Callable, Optional, Sequence, Tuple
import copy

from cv2 import log
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
        hidden_size: int = 256,
        soft_target_update_rate: Optional[float] = 0.005,
        learning_rate: float = 3e-4,
        # Actor-critic configuration
        num_actor_samples: int = 1,
        num_critic_updates: int = 1,
        # Settings for multiple critics
        num_critic_networks: int = 2,
        # Soft actor-critic
        temperature: float = 0.05,
        activation: str = 'tanh',
    ):
        super().__init__()

        self.actor = MLPPolicy(
            ac_dim=action_dim,
            ob_dim=np.prod(observation_shape),
            n_layers=n_layers,
            layer_size=hidden_size,
            activation=activation,
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.actor_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.999999)

        self.critics = nn.ModuleList(
            [
                StateActionCritic(
                    ob_dim=np.prod(observation_shape),
                    ac_dim=action_dim,
                    n_layers=n_layers,
                    size=hidden_size,
                    activation=activation,
                )
                for _ in range(num_critic_networks)
            ]
        )

        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=learning_rate)
        self.critic_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.999999)
        self.target_critics = nn.ModuleList(
            [
                StateActionCritic(
                    ob_dim=np.prod(observation_shape),
                    ac_dim=action_dim,
                    n_layers=n_layers,
                    size=hidden_size,
                    activation=activation,
                )
                for _ in range(num_critic_networks)
            ]
        )
        self.update_target_critic()

        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.discount = discount
        self.num_critic_networks = num_critic_networks
        self.temperature = temperature
        self.num_actor_samples = num_actor_samples
        self.num_critic_updates = num_critic_updates
        self.soft_target_update_rate = soft_target_update_rate

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
        with torch.no_grad():
            next_action_distribution: torch.distributions.Distribution = self.actor(next_obs)
            next_action = next_action_distribution.rsample()
            next_qs = self.target_critic(next_obs, next_action)

            next_qs_min = torch.min(next_qs, dim=0)[0]

            # next_action_entropy = self.entropy(next_action_distribution)
            # next_qs_min += self.temperature * next_action_entropy
            
            log_p = next_action_distribution.log_prob(next_action)
            next_qs_min -= self.temperature * log_p

            target_values: torch.Tensor = reward + (1 - done.float()) * self.discount * next_qs_min
            target_values = target_values.unsqueeze(0).expand(self.num_critic_networks, -1)

        # Predict Q-values
        q_values = self.critic(obs, action)

        # Compute loss
        loss: torch.Tensor = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            # "entropy_critic": next_action_entropy.mean().item(),
            "entropy_critic": -log_p.mean().item(),
        }

    def entropy(self, action_distribution: torch.distributions.Distribution):
        """
        Compute the (approximate) entropy of the action distribution for each batch element.
        """        
        actions = action_distribution.rsample((10,))
        log_probs = action_distribution.log_prob(actions)

        return -log_probs.mean(dim=0)

    def update_actor(self, obs: torch.Tensor):
        """
        Update the actor by one gradient step using REPARAMETRIZE.
        """
        # Sample from the actor
        action_distribution: torch.distributions.Distribution = self.actor(obs)
        action = action_distribution.rsample()

        q_values = self.critic(obs, action)
        q_values_min, _ = torch.min(q_values, axis=0)
        
        # entropy = self.entropy(action_distribution)
        # loss = -torch.mean(q_values_min - self.temperature * entropy)
        
        log_p = action_distribution.log_prob(action)
        loss = -torch.mean(q_values_min - self.temperature * log_p)

        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        # return {"actor_loss": loss.item(), "entropy": entropy.mean().item()}
        return {"actor_loss": loss.item(), "entropy": -log_p.mean().item()}

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
