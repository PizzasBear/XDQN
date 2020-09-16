from typing import Tuple, Union, List, Dict, overload, TypeVar, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from numpy import random

import per

LEARNING_RATE: float = 3e-4
REPLAY_CAPACITY: int = 2**19
DISCOUNT: float = 0.99
EPSILON_HIGH: float = 0.5
EPSILON_LOW: float = 0.1
EPSILON_DECAY: float = 8e-4
BATCH_SIZE: int = 64
MIN_REPLAY_LEN: int = 4_000
TARGET_UPDATE_INTERVAL: int = 1_000
PER_BETA_INCREMENT: float = 5e-4
NUM_QUANTILES: int = 50

T = TypeVar('T')
Device = Union[None, str, torch.device]


# noinspection PyAbstractClass
class MLP(nn.Module):
    __slots__ = 'layers', 'in_features', 'out_features'

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
            if isinstance(out_features, int):
                self.layers.append(nn.Linear(prev_features, out_features))
                prev_features = out_features
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
    __slots__ = 'layers',

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
    __slots__ = 'features_net', 'value_net', 'advantage_net'

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
class QuantileQNet(nn.Module):
    __slots__ = 'net', 'num_quantiles'
    net: nn.Module
    num_quantiles: int

    def __init__(self, net: nn.Module, num_quantiles: int):
        super().__init__()
        self.net = net
        self.num_quantiles = num_quantiles

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qs = self.net(x)
        return qs.view(-1,
                       qs.size()[1] // self.num_quantiles, self.num_quantiles)


# noinspection PyAbstractClass
class QuantileDuelingQNet(DuelingQNet):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: torch.Tensor = self.features_net(x)
        values: torch.Tensor = self.value_net(features).unsqueeze(
            1)  # -1, 1, Q
        num_quantiles: int = values.size()[2]
        advantages: torch.Tensor = self.advantage_net(features)  # -1, A*Q
        advantages = advantages.view(-1,
                                     advantages.size()[1] // num_quantiles,
                                     num_quantiles)  # -1, A, Q
        return values + advantages - advantages.mean(1, True)  # -1, A, Q


# noinspection PyAbstractClass
class Agent(nn.Module):
    __slots__ = 'net', 'target_net', 'buffer', 'opt', 'discount'

    net: nn.Module
    target_net: nn.Module
    buffer: per.PrioritisedReplayBuffer
    opt: optim.Optimizer

    discount: float

    def __init__(
        self,
        net: nn.Module,
        target_net: nn.Module,
        obs_shape: Union[int, Tuple[int, ...]],
        *,
        num_quantiles: int,
        lr: float = LEARNING_RATE,
        replay_cap: int = REPLAY_CAPACITY,
        obs_dtype: np.dtype = np.float32,
        discount: float = DISCOUNT,
        per_alpha: float = per.ALPHA,
        per_beta: float = per.BETA,
        reward_scaling: float = per.REWARD_SCALING,
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
        self.discount = discount
        tau0 = 0.5 / num_quantiles
        self.tau = nn.Parameter(torch.linspace(tau0, 1 - tau0,
                                               num_quantiles).view(1, -1, 1),
                                requires_grad=False)  # 1, Q, 1

    @torch.no_grad()
    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def store(self, prev_idx: int, q: float, obs: np.ndarray, action: int,
              reward: float, done: bool, next_obs: np.ndarray,
              device: Device) -> int:
        with torch.no_grad():
            if done:
                target_q = reward
            else:
                torch_next_obs = torch.tensor(next_obs,
                                              device=device).unsqueeze(0)
                next_q = self.target_net(torch_next_obs).squeeze(0).mean(
                    1).max(0).values.cpu().numpy()
                target_q = reward + self.discount * next_q
        return self.buffer.push(prev_idx, obs, action, reward, done,
                                target_q - q)

    def can_learn(self) -> bool:
        return MIN_REPLAY_LEN < len(self.buffer)

    def learn(self,
              t: int,
              writer: SummaryWriter,
              rng: random.Generator,
              device: Device,
              batch_size: int = BATCH_SIZE,
              n_step: int = 3):
        if not self.can_learn():
            return

        (obs, actions, rewards, masks, next_obs, weights,
         indices) = self.buffer.sample(rng, batch_size, self.discount, n_step)
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        masks = torch.tensor(masks, dtype=torch.bool, device=device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

        next_qs: torch.Tensor = self.net(next_obs)
        num_quantiles = next_qs.size()[2]
        next_actions: torch.Tensor = next_qs.mean(2,
                                                  True).argmax(1, True).expand(
                                                      -1, 1, num_quantiles)
        next_qs = self.target_net(next_obs).gather(1, next_actions)  # -1, 1, Q
        target_qs = rewards.view(-1, 1, 1) + masks.view(
            -1, 1, 1) * (self.discount**n_step) * next_qs  # -1, 1, Q
        qs = self.net(obs)  # -1, A, Q
        qs = qs.gather(1,
                       actions.view(-1, 1,
                                    1).expand(-1, 1, num_quantiles)).view(
                                        -1, num_quantiles, 1)  # -1, Q, 1
        error: torch.Tensor = qs - target_qs  # -1, Q, Q
        tau_weights = (self.tau -
                       (error.detach() < 0).float()).abs()  # -1, Q, Q
        priorities: np.ndarray = (error.detach().abs() * tau_weights).mean(
            (1, 2)).cpu().numpy()  # -1
        loss = (
            (F.smooth_l1_loss(qs.expand(-1, -1, num_quantiles),
                              target_qs.expand(-1, num_quantiles, -1),
                              reduction='none') * tau_weights).mean(2).sum(1) *
            weights).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.buffer.update_priorities(indices, priorities)
        writer.add_scalar('Data/Mean Abs Error', priorities.mean(), t)
        if not t % TARGET_UPDATE_INTERVAL:
            self.update_target()
        self.buffer.beta = min(1., self.buffer.beta + PER_BETA_INCREMENT)

    def net_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.net.state_dict()


# noinspection PyAbstractClass
class Actor:
    __slots__ = 'net', 'epsilon_decay', 'epsilon_low', 'epsilon_high'
    net: nn.Module
    epsilon_decay: float
    epsilon_low: float
    epsilon_high: float

    def __init__(
        self,
        net: nn.Module,
        epsilon_high: float = EPSILON_HIGH,
        epsilon_low: float = EPSILON_LOW,
        epsilon_decay: float = EPSILON_DECAY,
    ):
        self.net = net
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.epsilon_decay = epsilon_decay

    def get_epsilon(self, t: int):
        return (self.epsilon_high -
                self.epsilon_low) * self.epsilon_decay**t + self.epsilon_low

    def act(
        self,
        obs: np.ndarray,
        t: Optional[int],
        rng: Optional[random.Generator],
        device: Device = None,
        all_q: bool = False,
        epsilon: Optional[float] = None
    ) -> Tuple[int, Union[float, np.ndarray]]:
        with torch.no_grad():
            q: np.ndarray = self.net(
                torch.tensor(obs,
                             device=device).unsqueeze(0)).squeeze(0)  # A, Q
            q = q.mean(1).cpu().numpy()  # A
            action: Optional[int] = None
            if rng is not None:
                if epsilon is None and t is not None and 0 < self.epsilon_high:
                    epsilon = self.get_epsilon(t)
                if epsilon is not None and rng.random() < epsilon:
                    action = rng.integers(len(q))
            if action is None:
                action = q.argmax()
            return action, q if all_q else q[action]

    def load_state_dict(self,
                        state_dict: Dict[str, torch.Tensor],
                        strict: bool = True):
        self.net.load_state_dict(state_dict, strict)

    @overload
    def to(self: T,
           device: Optional[Union[int, torch.device]] = ...,
           dtype: Optional[Union[torch.dtype, str]] = ...,
           non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T,
           dtype: Union[torch.dtype, str],
           non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, tensor: torch.Tensor, non_blocking: bool = ...) -> T:
        ...

    def to(self, *args, **kwargs):
        self.net.to(*args, **kwargs)
        return self


def bellman_h(x: torch.Tensor, epsilon: float = 0.02) -> torch.Tensor:
    return torch.sign(x) * ((x.abs() + 1).sqrt() - 1) + epsilon * x


def bellman_ih(x: torch.Tensor, epsilon: float = 0.02) -> torch.Tensor:
    return torch.sign(x) * (((1 + 4 * epsilon *
                              (x.abs() + 1 + epsilon)).sqrt() - 1).square() *
                            (1 / (2 * epsilon))**2 - 1)
