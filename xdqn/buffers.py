from typing import Tuple, Union, Iterable
import numpy as np
from numpy import random
from sum_tree import SumTree
import torch
from xdqn.consts import *


class ActorBuffer:
    __slots__ = ('obs', 'actions', 'rewards', 'dones', 'len', 'cap')
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    len: int
    cap: int
    n_steps: int

    def __init__(self,
                 cap: int,
                 obs_shape: Union[int, Tuple[int, ...]],
                 obs_dtype: np.dtype = np.float32):
        if isinstance(obs_shape, int):
            obs_shape = obs_shape,
        self.obs = np.zeros((cap, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros(cap, dtype=np.int32)
        self.rewards = np.zeros(cap, dtype=np.float32)
        self.dones = np.zeros(cap, dtype=np.bool)
        self.cap = cap
        self.len = 0

    def push(self, obs: np.ndarray, action: int, reward: float, done: bool):
        i = self.len
        self.len += 1
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        return i

    def can_send(self) -> bool:
        return self.len == self.cap

    def clear(self):
        self.len = 0


class RecurrentActorBuffer(ActorBuffer):
    __slots__ = 'mem'
    mem: Tuple[np.ndarray, ...]

    def __init__(self,
                 cap: int,
                 obs_shape: Union[int, Tuple[int, ...]],
                 mem_shape: Tuple[Union[int, Tuple[int, ...]], ...],
                 obs_dtype: np.dtype = np.float32):
        super().__init__(cap, obs_shape, obs_dtype)
        mem = []
        for shape in mem_shape:
            if isinstance(shape, int):
                shape = shape,
            mem.append(np.zeros((cap, *shape), dtype=np.float32))
        self.mem = tuple(mem)

    def push(self,
             obs: np.ndarray,
             action: int,
             reward: float,
             done: bool,
             mem: Tuple[torch.Tensor, ...] = None):
        if mem is None:
            raise TypeError
        i = super().push(obs, action, reward, done)
        for j, m in enumerate(mem):
            self.mem[j][i] = m.cpu().numpy()


class ReplayBuffer:
    __slots__ = ('obs', 'actions', 'rewards', 'masks', 'reward_scaling', 'cap',
                 'lens', 'current_indices', 'n_steps', 'discount')
    # Buffers
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
    # Hyper Params
    reward_scaling: float
    cap: int
    n_steps: int
    discount: float
    # Buffer info
    lens: np.ndarray
    current_indices: np.ndarray

    def __init__(self,
                 num_envs: int,
                 cap: int,
                 obs_shape: Union[int, Tuple[int, ...]],
                 obs_dtype: np.dtype = np.float32,
                 reward_scaling: float = REWARD_SCALING,
                 n_steps: int = N_STEPS,
                 discount: float = DISCOUNT):
        if isinstance(obs_shape, int):
            obs_shape = obs_shape,
        self.obs = np.zeros((num_envs, cap, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((num_envs, cap), dtype=np.int32)
        self.rewards = np.zeros((num_envs, cap), dtype=np.float32)
        self.masks = np.zeros((num_envs, cap), dtype=np.bool)

        self.cap = cap
        self.n_steps = n_steps
        self.lens = np.zeros(num_envs, dtype=np.int32)
        self.current_indices = np.zeros(num_envs, dtype=np.int32)
        self.reward_scaling = reward_scaling
        self.discount = discount

    def load(self, env_id: int, buff: ActorBuffer):
        for i in range(buff.len):
            self.push(env_id, buff.obs[i], buff.actions[i], buff.rewards[i],
                      buff.dones[i])

    @property
    def num_envs(self):
        return len(self.lens)

    def __len__(self) -> int:
        return self.lens.sum()

    def next_index(self, env_id: int, idx: int) -> int:
        idx += 1
        idx %= self.lens[env_id]
        return idx

    def next_indices(self,
                     indices: Tuple[np.ndarray, np.ndarray],
                     steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        return indices[0], (indices[1] + steps) % self.lens[indices[0]]

    def push(self, env_id: int, obs: np.ndarray, action: int, reward: float,
             done: bool) -> int:
        if self.lens[env_id] < self.cap:
            self.lens[env_id] += 1
        idx = self.current_indices[env_id]
        self.current_indices[env_id] += 1
        self.current_indices[env_id] %= self.cap

        self.obs[env_id, idx] = obs
        self.actions[env_id, idx] = action
        reward /= self.reward_scaling
        self.rewards[env_id, idx] = reward
        for i in range(1, self.n_steps):
            if not self.masks[env_id, idx - i]:
                break
            reward *= self.discount
            self.rewards[env_id, idx - i] += reward
        mask = not done
        self.masks[env_id, idx] = mask
        for i in range(1, self.n_steps):
            self.masks[env_id, idx - i] &= mask
        return idx

    def get_data(
            self, indices: Tuple[np.ndarray,
                                 np.ndarray]) -> Tuple[np.ndarray, ...]:
        next_indices = self.next_indices(indices, self.n_steps)
        return (self.obs[indices], self.actions[indices],
                self.rewards[indices], self.masks[indices],
                self.obs[next_indices])

    def first_index(self, env_id: int) -> int:
        if self.lens[env_id] != self.cap:
            return 0
        else:
            return self.current_indices[env_id]

    def first_indices(self) -> np.ndarray:
        return np.where(self.lens != self.cap,
                        np.zeros(self.current_indices.shape, dtype=np.int32),
                        self.current_indices)

    def sample_indices(self,
                       rng: random.Generator,
                       batch_size: int,
                       ex_steps: int = 0):
        env_idx = rng.choice(self.num_envs,
                             batch_size,
                             p=self.lens / self.lens.sum())
        return (
            env_idx,
            (rng.integers(0, self.lens[env_idx] - self.n_steps - ex_steps) +
             self.first_indices()[env_idx]) % self.lens[env_idx])

    # noinspection DuplicatedCode
    def sample(self,
               rng: random.Generator,
               batch_size: int,
               ex_steps: int = 0):
        indices = self.sample_indices(rng, batch_size, ex_steps)
        return (*self.get_data(indices), indices)


class PrioritisedReplayBuffer(ReplayBuffer):
    __slots__ = 'priorities', 'max_priority', 'alpha', 'beta'
    # Buffers
    priorities: Tuple[SumTree]
    max_priority: float
    # Hyper Params
    alpha: float
    beta: float

    def __init__(self,
                 num_envs: int,
                 cap: int,
                 obs_shape: Union[int, Tuple[int, ...]],
                 obs_dtype: np.dtype = np.float32,
                 reward_scaling: float = REWARD_SCALING,
                 n_steps: int = N_STEPS,
                 discount: float = DISCOUNT,
                 alpha: float = PRIORITY_ALPHA,
                 beta: float = PRIORITY_BETA):
        if isinstance(obs_shape, int):
            obs_shape = obs_shape,
        super().__init__(num_envs, cap, obs_shape, obs_dtype, reward_scaling,
                         n_steps, discount)
        self.priorities = tuple(SumTree(cap) for _ in range(num_envs))
        self.max_priority = 5.

        self.alpha = alpha
        self.beta = beta

    def _get_priority(self, err, epsilon=0.1):
        return (np.abs(err) + epsilon)**self.alpha

    def push(self,
             env_id: int,
             obs: np.ndarray,
             action: int,
             reward: float,
             done: bool,
             err: float = None) -> int:
        if err is not None:
            priority = self._get_priority(err)
            self.max_priority = max(self.max_priority, priority)
        else:
            priority = self.max_priority
        idx = super().push(env_id, obs, action, reward, done)
        self.priorities[env_id][idx] = priority
        return idx

    def update_priorities(self, indices: Tuple[Iterable[int], Iterable[int]],
                          errs: Iterable[float]):
        for env_id, i, err in zip(indices[0], indices[1], errs):
            priority = self._get_priority(err)
            self.max_priority = max(self.max_priority, priority)
            self.priorities[env_id][i] = priority

    def total_priorities(self) -> float:
        return sum(priorities.total() for priorities in self.priorities)

    # noinspection PyUnboundLocalVariable
    def sample_indices(
        self,
        rng: random.Generator,
        batch_size: int,
        ex_steps: int = 0,
        get_weights: bool = True
    ) -> Union[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]], Tuple[
            np.ndarray, np.ndarray]]:
        steps = self.n_steps + ex_steps
        left_sum = [
            0. if self.current_indices[i] < steps else self.priorities[i].sum(
                0, self.current_indices[i] - steps)
            for i in range(self.num_envs)
        ]
        steps_sum = [
            self.priorities[i].sum(max(self.current_indices[i] - steps, 0),
                                   self.current_indices[i])
            for i in range(self.num_envs)
        ]
        buffer_probs = np.array([
            self.priorities[i].total() - steps_sum[i] -
            (self.priorities[i].sum(self.cap + self.current_indices[i] -
                                    steps, self.cap)
             if self.current_indices[i] < steps else 0.)
            for i in range(self.num_envs)
        ],
                                dtype=np.float32)
        total_priorities = buffer_probs.sum()
        env_indices = rng.choice(self.num_envs,
                                 batch_size,
                                 p=buffer_probs / total_priorities)
        indices = np.empty(batch_size, dtype=np.int32)
        if get_weights:
            weights = np.empty(batch_size, dtype=np.float32)
        for i in range(batch_size):
            env_id = env_indices[i]
            s = rng.uniform(0., buffer_probs[env_id])
            if left_sum[env_id] < s:
                s += steps_sum[env_id]
            idx, p = self.priorities[env_id].sample(s)
            assert 0 < p
            if get_weights:
                weights[i] = (len(self) * p / total_priorities)**-self.beta
            indices[i] = idx
        if get_weights:
            weights /= weights.max(initial=0.01)
            return weights, (env_indices, indices)
        else:
            return env_indices, indices

    def sample(self,
               rng: random.Generator,
               batch_size: int,
               ex_steps: int = 0):
        weights, indices = self.sample_indices(rng, batch_size, ex_steps)
        return (*self.get_data(indices), weights, indices)


class RecurrentPrioritisedExperienceReplay(PrioritisedReplayBuffer):
    __slots__ = 'mem'
    mem: Tuple[np.ndarray, ...]

    def __init__(self,
                 num_envs: int,
                 cap: int,
                 obs_shape: Union[int, Tuple[int, ...]],
                 mem_shape: Tuple[Union[int, Tuple[int, ...]], ...],
                 obs_dtype: np.dtype = np.float32,
                 reward_scaling: float = REWARD_SCALING,
                 n_steps: int = N_STEPS,
                 discount: float = DISCOUNT,
                 alpha: float = PRIORITY_ALPHA,
                 beta: float = PRIORITY_BETA):
        super().__init__(num_envs, cap, obs_shape, obs_dtype, reward_scaling,
                         n_steps, discount, alpha, beta)
        mem = []
        for shape in mem_shape:
            if isinstance(shape, int):
                shape = shape,
            mem.append(np.zeros((num_envs, cap, *shape), dtype=np.float32))
        self.mem = tuple(mem)

    def push(self,
             env_id: int,
             obs: np.ndarray,
             action: int,
             reward: float,
             done: bool,
             err: float = None,
             mem: Tuple[np.ndarray, ...] = None) -> int:
        if mem is None:
            raise TypeError
        idx = super().push(env_id, obs, action, reward, done, err)
        for i, m in enumerate(mem):
            self.mem[i][env_id, idx] = m
        return idx

    def load(self, env_id: int, buff: RecurrentActorBuffer):
        for i in range(buff.len):
            self.push(env_id,
                      buff.obs[i],
                      buff.actions[i],
                      buff.rewards[i],
                      buff.dones[i],
                      mem=tuple(m[i] for m in buff.mem))

    def update_memory(self, indices: np.ndarray, mem: Tuple[np.ndarray, ...]):
        for i, m in enumerate(mem):
            self.mem[i][indices] = m

    def get_data(self,
                 indices: Tuple[np.ndarray, np.ndarray],
                 get_mem: bool = True,
                 minimal: bool = False):
        next_indices = self.next_indices(indices, self.n_steps)
        if get_mem:
            mem = tuple(m[indices] for m in self.mem)
            next_mem = tuple(m[next_indices] for m in self.mem)
            if minimal:
                return mem, self.obs[indices], next_mem, self.obs[next_indices]
            else:
                return (mem, self.obs[indices], self.actions[indices],
                        self.rewards[indices], self.masks[indices], next_mem,
                        self.obs[next_indices])
        elif minimal:
            return self.obs[indices], self.obs[next_indices]
        else:
            return (self.obs[indices], self.actions[indices],
                    self.rewards[indices], self.masks[indices],
                    self.obs[next_indices])

    def sample(self,
               rng: random.Generator,
               batch_size: int,
               ex_steps: int = 0,
               get_mem: bool = True,
               minimal: bool = False):
        weights, indices = self.sample_indices(rng, batch_size, ex_steps)
        return (*self.get_data(indices, get_mem, minimal), weights, indices)
