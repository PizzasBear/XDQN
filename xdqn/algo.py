from typing import Tuple, Union, Dict, overload, TypeVar, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from numpy import random
import xdqn.buffers as buffers
from xdqn.nets import Device
from xdqn.consts import *

T = TypeVar('T')


# noinspection PyAbstractClass
class Agent(nn.Module):
    __slots__ = 'net', 'target_net', 'buffer', 'opt', 'transformed_bellman'

    net: nn.Module
    target_net: nn.Module
    buffer: buffers.RecurrentPrioritisedExperienceReplay
    opt: optim.Optimizer
    transformed_bellman: bool

    def __init__(
        self,
        net: nn.Module,
        target_net: nn.Module,
        num_envs: int,
        obs_shape: Union[int, Tuple[int, ...]],
        mem_shape: Tuple[Union[int, Tuple[int, ...]], ...],
        *,
        num_quantiles: int,
        lr: float = LEARNING_RATE,
        replay_cap: int = REPLAY_CAPACITY,
        obs_dtype: np.dtype = np.float32,
        discount: float = DISCOUNT,
        n_steps: int = N_STEPS,
        per_alpha: float = PRIORITY_ALPHA,
        per_beta: float = PRIORITY_BETA,
        reward_scaling: float = REWARD_SCALING,
        transformed_bellman: bool = TRANSFORMED_BELLMAN,
    ):
        super().__init__()
        self.net = net
        self.target_net = target_net
        self.update_target()
        self.buffer = buffers.RecurrentPrioritisedExperienceReplay(
            num_envs,
            replay_cap // num_envs,
            obs_shape,
            mem_shape,
            obs_dtype=obs_dtype,
            reward_scaling=reward_scaling,
            n_steps=n_steps,
            discount=discount,
            alpha=per_alpha,
            beta=per_beta)
        self.opt = optim.Adam(net.parameters(), lr=lr)
        self.transformed_bellman = transformed_bellman
        tau0 = 0.5 / num_quantiles
        self.tau = nn.Parameter(torch.linspace(tau0, 1 - tau0,
                                               num_quantiles).view(1, -1, 1),
                                requires_grad=False)  # 1, Q, 1

    @torch.no_grad()
    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def store(self, env_id: int, mem: Tuple[np.ndarray, ...], obs: np.ndarray,
              action: int, reward: float, done: bool) -> int:
        return self.buffer.push(env_id, obs, action, reward, done, mem=mem)

    def can_learn(self) -> bool:
        return MIN_REPLAY_LEN < len(self.buffer)

    # noinspection PyTypeChecker
    def learn(self,
              t: int,
              writer: SummaryWriter,
              rng: random.Generator,
              device: Device,
              batch_size: int = BATCH_SIZE,
              recurrent_steps: int = RECURRENT_STEPS,
              burn_in_steps: int = BURN_IN_STEPS):
        if not self.can_learn():
            return

        actions, rewards, masks = None, None, None
        if burn_in_steps:
            (mem, obs, target_mem, next_obs, weights,
             indices) = self.buffer.sample(rng,
                                           batch_size,
                                           minimal=True,
                                           ex_steps=recurrent_steps +
                                           burn_in_steps)
        else:
            (mem, obs, actions, rewards, masks, target_mem, next_obs, weights,
             indices) = self.buffer.sample(rng,
                                           batch_size,
                                           minimal=True,
                                           ex_steps=recurrent_steps +
                                           burn_in_steps)
        o_indices = indices
        mean_priorities = np.zeros_like(weights)
        max_priorities = np.zeros_like(weights)
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        mem = tuple(
            torch.tensor(m, dtype=torch.float32, device=device) for m in mem)
        target_mem = tuple(
            torch.tensor(m, dtype=torch.float32, device=device)
            for m in target_mem)
        with torch.no_grad():
            for i in reversed(range(burn_in_steps)):
                "Move data to PyTorch"
                obs = torch.tensor(obs, dtype=torch.float32, device=device)
                next_obs = torch.tensor(next_obs,
                                        dtype=torch.float32,
                                        device=device)
                "Remember"
                mem = self.net.remember(obs, mem)
                target_mem = self.target_net.remember(next_obs, target_mem)
                "Update Replay Buffer"
                self.buffer.update_memory(
                    self.buffer.next_indices(indices, self.buffer.n_steps),
                    tuple(m.cpu().numpy() for m in target_mem))
                "Get next data"
                indices = self.buffer.next_indices(indices)
                if i:
                    obs, next_obs = self.buffer.get_data(indices, False, True)
                else:
                    obs, actions, rewards, masks, next_obs = self.buffer.get_data(
                        indices, False)
        loss = 0.
        for i in reversed(range(recurrent_steps)):
            "Move data to PyTorch"
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
            actions = torch.tensor(actions, dtype=torch.int64, device=device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            masks = torch.tensor(masks, dtype=torch.bool, device=device)
            next_obs = torch.tensor(next_obs,
                                    dtype=torch.float32,
                                    device=device)
            with torch.no_grad():
                "Calculate next actions"
                next_qs, _ = self.net(next_obs, target_mem)
                if self.transformed_bellman:
                    next_qs = bellman_ih(next_qs)
                num_quantiles = next_qs.size()[2]
                next_actions = next_qs.mean(2, True).argmax(1, True).expand(
                    -1, 1, num_quantiles)
                "Calculate next qs"
                next_qs, target_mem = self.target_net(next_obs, target_mem)
                next_qs = next_qs.gather(1, next_actions)  # -1, 1, Q
                if self.transformed_bellman:
                    next_qs = bellman_ih(next_qs)
                "Calculate target qs"
                target_qs = rewards.view(-1, 1, 1) + masks.view(
                    -1, 1, 1) * (self.buffer.discount **
                                 self.buffer.n_steps) * next_qs  # -1, 1, Q
                if self.transformed_bellman:
                    target_qs = bellman_h(target_qs)  # rescale qs, (-1, 1, Q)
            "Calculate current qs"
            qs, mem = self.net(obs, mem)  # -1, A, Q
            mem = tuple(
                torch.where(masks, m, torch.zeros_like(m)) for m in mem)
            actions = actions.view(-1, 1, 1).expand(-1, 1, num_quantiles)
            qs = qs.gather(1, actions).view(-1, num_quantiles, 1)  # -1, Q, 1
            "Calculate Loss"
            error: torch.Tensor = qs - target_qs  # -1, Q, Q
            tau_weights = (self.tau -
                           (error.detach() < 0).float()).abs()  # -1, Q, Q
            loss = ((F.smooth_l1_loss(qs.expand(-1, -1, num_quantiles),
                                      target_qs.expand(-1, num_quantiles, -1),
                                      reduction='none') *
                     tau_weights).mean(2).sum(1) * weights).mean() + loss
            with torch.no_grad():
                "Calculate priorities (eg. Max and Mean)"
                current_priorities = (error.abs() * tau_weights).mean(
                    (1, 2)).cpu().numpy()  # -1
                mean_priorities += current_priorities
                max_priorities = np.maximum(max_priorities, current_priorities)
                if i:
                    "Update Replay Buffer"
                    self.buffer.update_memory(
                        self.buffer.next_indices(indices, self.buffer.n_steps),
                        tuple(m.cpu().numpy() for m in target_mem))
                    indices = self.buffer.next_indices(indices)
                    obs, actions, rewards, masks, next_obs = self.buffer.get_data(
                        indices, False)
        mean_priorities /= recurrent_steps
        priorities = PRIORITY_MEAN_MAX * max_priorities + (
            1 - PRIORITY_MEAN_MAX) * mean_priorities
        priorities: np.ndarray
        assert np.all(np.isfinite(priorities))
        self.buffer.update_priorities(o_indices, priorities)

        loss /= recurrent_steps
        assert loss.isfinite()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        writer.add_scalar('Data/Mean Priorities', priorities.mean(), t)
        if not t % TARGET_UPDATE_INTERVAL:
            self.update_target()
        # self.buffer.beta = min(1., self.buffer.beta + PER_BETA_INCREMENT)

    def load(self, env_id: int, buff: buffers.RecurrentActorBuffer):
        self.buffer.load(env_id, buff)

    def net_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.net.state_dict()

    def mem_size(self) -> Tuple[int, ...]:
        return self.net.mem_size()

    def init_mem(self,
                 batch_size: int,
                 device: Device = None) -> Tuple[torch.Tensor, ...]:
        return self.net.init_mem(batch_size, device)


class DecayedEpsilonGreedy:
    __slots__ = 'epsilon_decay', 'epsilon_low', 'epsilon0'
    epsilon_decay: float
    epsilon_low: float
    epsilon0: float

    def __init__(
        self,
        epsilon_high: float = EPSILON_HIGH,
        epsilon_low: float = EPSILON_LOW,
        epsilon_decay: float = EPSILON_DECAY,
    ):
        self.epsilon_low = epsilon_low
        self.epsilon_decay = epsilon_decay
        self.epsilon0 = epsilon_high - epsilon_low

    def update(self):
        self.epsilon0 = self.epsilon0 * self.epsilon_decay

    @property
    def epsilon(self):
        return self.epsilon0 + self.epsilon_low


# noinspection PyAbstractClass
class Actor:
    __slots__ = 'net', 'transformed_bellman'
    net: nn.Module
    transformed_bellman: bool

    def __init__(self,
                 net: nn.Module,
                 transformed_bellman: bool = TRANSFORMED_BELLMAN):
        self.net = net
        self.transformed_bellman = transformed_bellman

    @torch.no_grad()
    def act(
        self,
        mem: Tuple[Union[np.ndarray, torch.Tensor], ...],
        obs: np.ndarray,
        rng: Optional[random.Generator],
        device: Device = None,
        get_q: bool = False,
        epsilon: Optional[float] = None,
        numpy_mem: bool = False,
    ) -> Union[Tuple[int, Tuple[Union[np.ndarray, torch.Tensor], ...]], Tuple[
            int, np.ndarray, Tuple[Union[np.ndarray, torch.Tensor], ...]]]:
        obs = torch.tensor(obs, device=device).unsqueeze(0)
        mem = tuple(m if isinstance(m, torch.Tensor) else torch.
                    tensor(m, device=device) for m in mem)
        q, mem = self.net(obs, mem)  # A, Q
        q.squeeze_(0)
        q = bellman_ih(q)
        q = q.mean(1).cpu().numpy()  # A
        action: Optional[int] = None
        if rng is not None:
            if epsilon is not None and rng.random() < epsilon:
                action = rng.integers(len(q))
        if action is None:
            action = q.argmax()
        if numpy_mem:
            mem = tuple(m.cpu().numpy() for m in mem)
        if get_q:
            return action, q, mem
        else:
            return action, mem

    def load_state_dict(self,
                        state_dict: Dict[str, torch.Tensor],
                        strict: bool = True):
        self.net.load_state_dict(state_dict, strict)

    def mem_size(self) -> Tuple[int, ...]:
        return self.net.mem_size()

    def init_mem(self,
                 batch_size: int,
                 device: Device = None) -> Tuple[torch.Tensor, ...]:
        return self.net.init_mem(batch_size, device)

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
