from typing import Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import numpy as np
from numpy import random
import per

LEARNING_RATE: float = 2e-4
REPLAY_CAPACITY: int = 2**19
DISCOUNT: float = 0.99
EPSILON_GREEDY: float = 0.1
BATCH_SIZE: int = 64


# noinspection PyAbstractClass
class MLP(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    layers: nn.ModuleList

    def __init__(self,
                 layer_sizes: List[int],
                 in_features: int,
                 activation: nn.Module = None,
                 out_features: Union[None, int, Tuple[int, nn.Module]] = None):
        super().__init__()
        if activation is None:
            activation = nn.ReLU()

        self.in_features = in_features
        self.layers = nn.ModuleList()
        prev_features = in_features
        for size in layer_sizes:
            layer = nn.Linear(prev_features, size)
            init.orthogonal_(layer.weight)
            # init.zeros_(layer.bias)
            self.layers.append(layer)
            self.layers.append(activation)
            prev_features = size

        if out_features is not None:
            prev_features = out_features
            if isinstance(out_features, int):
                self.layers.append(nn.Linear(prev_features, out_features))
            else:
                self.layers.append(nn.Linear(prev_features, out_features[0]))
                self.layers.append(out_features[1])
                prev_features, _ = prev_features
        self.out_features = prev_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# noinspection PyAbstractClass
class ConvNet(nn.Module):
    layers: nn.ModuleList

    def __init__(self,
                 layers: List[Union[Tuple[int, int, int], Tuple[int, int]]],
                 in_channels: int,
                 flatten: bool = False):
        super().__init__()
        prev_channels = in_channels
        self.layers = nn.ModuleList()
        for layer in layers:
            if len(layer) == 3:
                channels, kernel, stride = layer
            else:
                channels, kernel = layer
                stride = 1
            layer = nn.Conv2d(prev_channels, channels, kernel, stride)
            init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            self.layers.append(layer)

            self.layers.append(nn.ReLU())
            prev_channels = channels
        if flatten:
            self.layers.append(nn.Flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# noinspection PyAbstractClass
class DuelingQNet(nn.Module):
    features_net: nn.Module
    value_net: nn.Module
    advantage_net: nn.Module

    def __init__(self, feature_net, value_net, advantage_net):
        super().__init__()
        self.features_net = feature_net
        self.value_net = value_net
        self.advantage_net = advantage_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: torch.Tensor = self.features_net(x)
        advantages: torch.Tensor = self.advantage_net(features)
        return self.value_net(features) + advantages - advantages.mean(1, True)


# noinspection PyAbstractClass
class Actor(nn.Module):
    net: nn.Module

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def act(self, obs: np.ndarray,
            device: Union[None, str, torch.device]) -> Tuple[int, float]:
        with torch.no_grad():
            q: np.ndarray = self.net(
                torch.tensor(
                    obs, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
            action: int = q.argmax()
            return action, q[action]


# noinspection PyAbstractClass
class Agent(nn.Module):
    net: nn.Module
    target_net: nn.Module
    buffer: per.PrioritisedReplayBuffer
    opt: optim.Optimizer

    DISCOUNT: float

    def __init__(
        self,
        net: nn.Module,
        target_net: nn.Module,
        obs_shape: Union[int, Tuple[int, ...]],
        *,
        lr: float = LEARNING_RATE,
        replay_cap: int = REPLAY_CAPACITY,
        obs_dtype: np.dtype = np.float32,
        discount: float = DISCOUNT,
        per_alpha: float = per.ALPHA,
        per_beta: float = per.BETA,
        reward_scaling: float = per.REWARD_SCALING,
        epsilon_greedy: float = EPSILON_GREEDY,
    ):
        super().__init__()
        self.net = net
        self.target_net = target_net
        self.update_target()
        self.buffer = per.PrioritisedReplayBuffer(
            replay_cap,
            obs_shape,
            obs_dtype=obs_dtype,
            alpha=per_alpha,
            beta=per_beta,
            reward_scaling=reward_scaling)
        self.opt = optim.Adam(net.parameters(), lr=lr)
        self.DISCOUNT = discount

    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def store(self, prev_idx: int, q: float, obs: np.ndarray, action: int,
              reward: float, done: bool, next_obs: np.ndarray,
              device: Union[None, str, torch.device]) -> int:
        with torch.no_grad():
            target_q = (reward + self.DISCOUNT * self.target_net(
                torch.tensor(next_obs, device=device).unsqueeze(0)).max(
                    1).unsqueeze(0).cpu().numpy() if not done else reward)
        return self.buffer.push(prev_idx, target_q - q, obs, action, reward,
                                done)

    def learn(self, rng: random.Generator, batch_size: int = BATCH_SIZE):
        states, actions, rewards, masks, next_states = self.buffer.sample(
            rng, batch_size)
