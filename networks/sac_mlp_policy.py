from typing import Optional, Union

from torch import nn

import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


def make_tanh_transformed(
    mean: torch.Tensor, std: Union[float, torch.Tensor]
) -> distributions.Distribution:
    if isinstance(std, float):
        std = torch.tensor(std, device=mean.device)

    if std.shape == ():
        std = std.expand(mean.shape)

    return distributions.Independent(
        distributions.TransformedDistribution(
            base_distribution=distributions.Normal(mean, std),
            transforms=[distributions.TanhTransform(cache_size=1)],
        ),
        reinterpreted_batch_ndims=1,
    )


class MLPPolicy(nn.Module):
    """
    Base MLP policy, which can take an observation and output a distribution over actions.

    This class implements `forward()` which takes a (batched) observation and returns a distribution over actions.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        activation: str='tanh'
    ):
        super().__init__()

        self.net = ptu.build_mlp(
            input_size=ob_dim,
            output_size=2*ac_dim,
            n_layers=n_layers,
            size=layer_size,
            activation=activation
        ).to(ptu.device)

    def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
        mean, std = torch.chunk(self.net(obs), 2, dim=-1)
        std = torch.nn.functional.softplus(std) + 1e-6

        action_distribution = make_tanh_transformed(mean, std)
        return action_distribution
 
