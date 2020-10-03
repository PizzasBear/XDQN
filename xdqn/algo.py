from typing import Tuple, Union, Dict, overload, TypeVar, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from numpy import random
import xdqn.buffers as buffers
from xdqn.nets import Device, QuantileDuelingQNet
from xdqn.consts import *

T = TypeVar('T')


# noinspection PyAbstractClass
class Agent(nn.Module):
    __slots__ = 'net', 'target_net', 'buffer', 'opt', 'transformed_bellman', 'frame_stacking'
    net: QuantileDuelingQNet
    target_net: Optional[QuantileDuelingQNet]
    buffer: Optional[buffers.RecurrentPrioritisedExperienceReplay]
    opt: Optional[optim.Optimizer]
    transformed_bellman: bool
    frame_stacking: Optional[int]

    def __init__(
        self,
        net: QuantileDuelingQNet,
        target_net: Optional[QuantileDuelingQNet],
        num_envs: int,
        obs_shape: Union[int, Tuple[int, ...]],
        mem_shape: Tuple[Union[int, Tuple[int, ...]], ...],
        *,
        train: bool = True,
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
        compress_fn: buffers.TransformFunction = None,
        decompress_fn: buffers.TransformFunction = None,
        frame_stacking: Optional[int] = None,
    ):
        super().__init__()
        self.net = net
        if train:
            self.target_net = target_net
            self.update_target()
        self.buffer = buffers.RecurrentPrioritisedExperienceReplay(
            num_envs,
            replay_cap // num_envs,
            obs_shape,
            mem_shape,
            compress_fn=compress_fn,
            decompress_fn=decompress_fn,
            obs_dtype=obs_dtype,
            reward_scaling=reward_scaling,
            n_steps=n_steps,
            discount=discount,
            alpha=per_alpha,
            beta=per_beta) if train else None
        self.opt = optim.Adam(net.parameters(), lr=lr) if train else None
        self.transformed_bellman = transformed_bellman
        tau0 = 0.5 / num_quantiles
        self.tau = nn.Parameter(torch.linspace(tau0, 1 - tau0,
                                               num_quantiles).view(1, -1, 1),
                                requires_grad=False)  # 1, Q, 1
        self.frame_stacking = frame_stacking

    @torch.no_grad()
    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def store(self, env_id: int, mem: List[torch.Tensor], obs: np.ndarray,
              action: int, reward: float, done: bool):
        self.buffer.push(env_id,
                         obs,
                         action,
                         reward,
                         done,
                         mem=[m.cpu().numpy() for m in mem])

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

        (mem, obs, actions, mem_masks, rewards, masks, target_mem, next_obs,
         next_mem_masks, weights, indices) = self.buffer.sample(
             rng,
             batch_size,
             ex_steps=recurrent_steps + burn_in_steps,
             frame_stacking=self.frame_stacking)
        o_indices = indices
        mean_priorities = np.zeros_like(weights)
        max_priorities = np.zeros_like(weights)
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        mem = tuple(
            torch.tensor(m, dtype=torch.float32, device=device) for m in mem)
        target_mem = tuple(
            torch.tensor(m, dtype=torch.float32, device=device)
            for m in target_mem)
        init_mem = self.net.get_init_mem(batch_size)
        init_target_mem = [
            m.detach() for m in self.target_net.get_init_mem(batch_size)
        ]
        for _ in range(burn_in_steps):
            "Move data to PyTorch"
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
            mem_masks = torch.tensor(mem_masks,
                                     dtype=torch.bool,
                                     device=device)
            next_obs = torch.tensor(next_obs,
                                    dtype=torch.float32,
                                    device=device)
            next_mem_masks = torch.tensor(next_mem_masks,
                                          dtype=torch.bool,
                                          device=device)
            "Remember"
            mem = mask_mem(mem_masks, self.net.remember(obs, mem), init_mem)
            with torch.no_grad():
                target_mem = mask_mem(
                    next_mem_masks,
                    self.target_net.remember(next_obs, target_mem),
                    init_target_mem)
            "Update the replay buffer"
            indices = self.buffer.next_indices(indices)
            self.buffer.update_memory(
                self.buffer.next_indices(indices, self.buffer.n_steps),
                tuple(m.cpu().numpy() for m in target_mem))
            "Get next data"
            (obs, actions, mem_masks, rewards, masks, next_obs,
             next_mem_masks) = self.buffer.get_data(
                 indices, False, frame_stacking=self.frame_stacking)
        loss = 0.
        for i in reversed(range(recurrent_steps)):
            "Move data to PyTorch"
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
            actions = torch.tensor(actions, dtype=torch.int64, device=device)
            mem_masks = torch.tensor(mem_masks,
                                     dtype=torch.bool,
                                     device=device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            masks = torch.tensor(masks, dtype=torch.bool, device=device)
            next_obs = torch.tensor(next_obs,
                                    dtype=torch.float32,
                                    device=device)
            next_mem_masks = torch.tensor(next_mem_masks,
                                          dtype=torch.bool,
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
                target_mem = mask_mem(next_mem_masks, target_mem,
                                      init_target_mem)
                next_qs = next_qs.gather(1, next_actions)  # -1, 1, Q
                if self.transformed_bellman:
                    next_qs = bellman_ih(next_qs)
                "Calculate target qs"
                target_qs = rewards.view(-1, 1, 1) + masks.view(
                    -1, 1, 1) * (self.buffer.discount**
                                 self.buffer.n_steps) * next_qs  # -1, 1, Q
                if self.transformed_bellman:
                    target_qs = bellman_h(target_qs)  # rescale qs, (-1, 1, Q)
            "Calculate current qs"
            qs, mem = self.net(obs, mem)  # -1, A, Q
            mem = mask_mem(mem_masks, mem, init_mem)
            actions = actions.view(-1, 1, 1).expand(-1, 1, num_quantiles)
            qs = qs.gather(1, actions).view(-1, num_quantiles, 1)  # -1, Q, 1
            "Calculate loss"
            error: torch.Tensor = qs - target_qs  # -1, Q, Q
            tau_weights = (self.tau -
                           (error.detach() < 0).float()).abs()  # -1, Q, Q
            loss = ((F.smooth_l1_loss(qs.expand(-1, -1, num_quantiles),
                                      target_qs.expand(-1, num_quantiles, -1),
                                      reduction='none') *
                     tau_weights).mean(2).sum(1) * weights).mean() + loss
            with torch.no_grad():
                "Calculate priorities (their mean and max)"
                current_priorities = (error.abs() * tau_weights).mean(
                    (1, 2)).cpu().numpy()  # -1
                mean_priorities += current_priorities
                max_priorities = np.maximum(max_priorities, current_priorities)
                if i:
                    "Update Replay Buffer"
                    indices = self.buffer.next_indices(indices)
                    self.buffer.update_memory(
                        self.buffer.next_indices(indices, self.buffer.n_steps),
                        [m.cpu().numpy() for m in target_mem])
                    "Get next data"
                    (obs, actions, mem_masks, rewards, masks, next_obs,
                     next_mem_masks) = self.buffer.get_data(
                         indices, False, frame_stacking=self.frame_stacking)
        "Update Priorities"
        mean_priorities /= recurrent_steps
        priorities = PRIORITY_MEAN_MAX * max_priorities + (
            1 - PRIORITY_MEAN_MAX) * mean_priorities
        priorities: np.ndarray
        assert np.all(np.isfinite(priorities))
        self.buffer.update_priorities(o_indices, priorities)

        "Update Parameters"
        loss /= recurrent_steps
        assert loss.isfinite()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        "Update Target Net"
        writer.add_scalar('Data/Mean Priorities', priorities.mean(), t)
        if not t % TARGET_UPDATE_INTERVAL:
            self.update_target()

    def net_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.net.state_dict()

    def mem_size(self) -> Tuple[int, ...]:
        return self.net.mem_size()

    def get_init_mem(self,
                     batch_size: int,
                     detached: bool = True) -> List[torch.Tensor]:
        init_mem = self.net.get_init_mem(batch_size)
        return [m.detach() for m in init_mem] if detached else init_mem

    @torch.no_grad()
    def act(
        self,
        mem: List[torch.Tensor],
        obs: np.ndarray,
        rng: Optional[random.Generator],
        device: Device = None,
        get_q: bool = False,
        epsilon: np.ndarray = None,
    ) -> Union[Tuple[int, List[torch.Tensor]], Tuple[int, np.ndarray, Tuple[
            Union[np.ndarray, torch.Tensor], ...]]]:
        num = obs.shape[0]
        if epsilon is None:
            epsilon = np.zeros(num, dtype=np.float32)
        obs = torch.tensor(obs, device=device)
        q, mem = self.net(obs, mem)  # A, Q
        if TRANSFORMED_BELLMAN:
            q = bellman_ih(q)
        q = q.mean(2).cpu().numpy()  # A
        if rng is not None:
            # if 0 < epsilon and rng.random() < epsilon:
            #     action = rng.integers(len(q))
            is_rand: np.ndarray
            is_rand = rng.random(num) < epsilon
            action = np.empty(num, dtype=np.int32)
            action[is_rand] = rng.integers(q.shape[1],
                                           size=np.count_nonzero(is_rand))
            is_not_rand = ~is_rand
            action[is_not_rand] = q[is_not_rand].argmax(1)
        else:
            action = q.argmax(1)
        if get_q:
            return action, q, mem
        else:
            return action, mem


def mask_mem(masks, mem, init_mem):
    return [
        torch.where(masks.unsqueeze(1), m, im) for m, im in zip(mem, init_mem)
    ]


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


#
#
# # noinspection PyAbstractClass
# class Actor:
#     __slots__ = 'net', 'transformed_bellman'
#     net: QuantileDuelingQNet
#     transformed_bellman: bool
#
#     def __init__(self,
#                  net: QuantileDuelingQNet,
#                  transformed_bellman: bool = TRANSFORMED_BELLMAN):
#         self.net = net
#         self.transformed_bellman = transformed_bellman
#
#     @torch.no_grad()
#     def act(
#         self,
#         mem: Tuple[Union[np.ndarray, torch.Tensor], ...],
#         obs: np.ndarray,
#         rng: Optional[random.Generator],
#         device: Device = None,
#         get_q: bool = False,
#         epsilon: Optional[float] = None,
#         numpy_mem: bool = False,
#     ) -> Union[Tuple[int, Tuple[Union[np.ndarray, torch.Tensor], ...]], Tuple[
#             int, np.ndarray, Tuple[Union[np.ndarray, torch.Tensor], ...]]]:
#         obs = torch.tensor(obs, device=device).unsqueeze(0)
#         mem = tuple(m if isinstance(m, torch.Tensor) else torch.
#                     tensor(m, device=device) for m in mem)
#         q, mem = self.net(obs, mem)  # A, Q
#         q.squeeze_(0)
#         q = bellman_ih(q)
#         q = q.mean(1).cpu().numpy()  # A
#         action: Optional[int] = None
#         if rng is not None:
#             if epsilon is not None and rng.random() < epsilon:
#                 action = rng.integers(len(q))
#         if action is None:
#             action = q.argmax()
#         if numpy_mem:
#             mem = tuple(m.cpu().numpy() for m in mem)
#         if get_q:
#             return action, q, mem
#         else:
#             return action, mem
#
#     def load_state_dict(self,
#                         state_dict: Dict[str, torch.Tensor],
#                         strict: bool = True):
#         self.net.load_state_dict(state_dict, strict)
#
#     def mem_size(self) -> Tuple[int, ...]:
#         return self.net.mem_size()
#
#     @torch.no_grad()
#     def get_init_mem(self) -> Tuple[torch.Tensor, ...]:
#         return tuple(m.detach() for m in self.net.get_init_mem(1))
#
#     @overload
#     def to(self: T,
#            device: Optional[Union[int, torch.device]] = ...,
#            dtype: Optional[Union[torch.dtype, str]] = ...,
#            non_blocking: bool = ...) -> T:
#         ...
#
#     @overload
#     def to(self: T,
#            dtype: Union[torch.dtype, str],
#            non_blocking: bool = ...) -> T:
#         ...
#
#     @overload
#     def to(self: T, tensor: torch.Tensor, non_blocking: bool = ...) -> T:
#         ...
#
#     def to(self, *args, **kwargs):
#         self.net.to(*args, **kwargs)
#         return self


def bellman_h(x: torch.Tensor, epsilon: float = 0.02) -> torch.Tensor:
    return torch.sign(x) * ((x.abs() + 1).sqrt() - 1) + epsilon * x


def bellman_ih(x: torch.Tensor, epsilon: float = 0.02) -> torch.Tensor:
    return torch.sign(x) * (((1 + 4 * epsilon *
                              (x.abs() + 1 + epsilon)).sqrt() - 1).square() *
                            (1 / (2 * epsilon))**2 - 1)
