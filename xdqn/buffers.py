from typing import Tuple, Union, Iterable, Optional, Callable, List
import numpy as np
from numpy import random
from sum_tree import SumTree
# import torch
from xdqn.consts import *

TransformFunction = Callable[[np.ndarray], np.ndarray]


class FrameStack:
    __slots__ = 'frames', 'frame_stacking'
    frames: np.ndarray
    frame_stacking: Optional[int]

    def __init__(self,
                 first_obs: np.ndarray,
                 frame_stacking: Optional[int] = None):
        self.frame_stacking = frame_stacking
        if frame_stacking is None:
            self.frames = first_obs
        else:
            self.frames = np.repeat(np.expand_dims(first_obs, 0),
                                    frame_stacking, 0)

    def update(self, done: bool, next_obs: np.ndarray):
        if self.frame_stacking is None:
            self.frames = next_obs
        elif done:
            self.frames[:] = next_obs
        else:
            self.frames[0] = next_obs
            self.frames = np.roll(self.frames, -1, 0)


class VecFrameStack:
    __slots__ = 'frames', 'frame_stacking'
    frames: np.ndarray
    frame_stacking: Optional[int]

    def __init__(self,
                 first_obs: np.ndarray,
                 frame_stacking: Optional[int] = None):
        self.frame_stacking = frame_stacking
        if frame_stacking is None:
            self.frames = first_obs
        else:
            self.frames = np.repeat(np.expand_dims(first_obs, 1),
                                    frame_stacking, 1)

    def update(self, dones: np.ndarray, next_obs: np.ndarray):
        if self.frame_stacking is None:
            self.frames = next_obs
        else:
            self.frames[dones] = np.expand_dims(next_obs[dones], axis=1)
            not_dones = ~dones
            self.frames[not_dones, 0] = next_obs[not_dones]
            self.frames[not_dones] = np.roll(self.frames[not_dones], -1, 1)


class ReplayBuffer:
    __slots__ = ('obs', 'actions', 'rewards', 'masks', 'multi_masks', 'reward_scaling', 'cap',
                 'lens', 'current_indices', 'n_steps', 'discount',
                 'compress_fn', 'decompress_fn')
    # Buffers
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
    multi_masks: np.ndarray
    # Hyper Params
    reward_scaling: float
    cap: int
    n_steps: int
    discount: float
    # Buffer info
    lens: np.ndarray
    current_indices: np.ndarray
    compress_fn: TransformFunction
    decompress_fn: TransformFunction

    def __init__(self,
                 num_envs: int,
                 cap: int,
                 obs_shape: Union[int, Tuple[int, ...]],
                 obs_dtype: np.dtype = np.float32,
                 reward_scaling: float = REWARD_SCALING,
                 n_steps: int = N_STEPS,
                 discount: float = DISCOUNT,
                 compress_fn: TransformFunction = None,
                 decompress_fn: TransformFunction = None):
        if isinstance(obs_shape, int):
            obs_shape = obs_shape,
        self.obs = np.zeros((num_envs, cap, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((num_envs, cap), dtype=np.int32)
        self.rewards = np.zeros((num_envs, cap), dtype=np.float32)
        self.masks = np.zeros((num_envs, cap), dtype=np.bool)
        self.multi_masks = np.zeros((num_envs, cap), dtype=np.bool)

        self.cap = cap
        self.n_steps = n_steps
        self.lens = np.zeros(num_envs, dtype=np.int32)
        self.current_indices = np.zeros(num_envs, dtype=np.int32)
        self.reward_scaling = reward_scaling
        self.discount = discount
        self.compress_fn = (
            lambda x: x) if compress_fn is None else compress_fn
        self.decompress_fn = (
            lambda x: x) if decompress_fn is None else decompress_fn

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

    def push(self,
             env_id: int,
             obs: np.ndarray,
             action: int,
             reward: float,
             done: bool,
             compress: bool = True) -> int:
        if self.lens[env_id] < self.cap:
            self.lens[env_id] += 1
        idx = self.current_indices[env_id]
        self.current_indices[env_id] += 1
        self.current_indices[env_id] %= self.cap

        self.obs[env_id, idx] = self.compress_fn(obs) if compress else obs
        self.actions[env_id, idx] = action
        reward /= self.reward_scaling
        self.rewards[env_id, idx] = reward
        for i in range(1, self.n_steps):
            if not self.multi_masks[env_id, idx - i]:
                break
            reward *= self.discount
            self.rewards[env_id, idx - i] += reward
        mask = not done
        self.masks[env_id, idx] = mask
        self.multi_masks[env_id, idx] = mask
        for i in range(1, self.n_steps):
            self.multi_masks[env_id, idx - i] &= mask
        return idx

    def get_obs(
        self,
        indices: Tuple[np.ndarray, np.ndarray],
        frame_stacking: Optional[int] = None
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if frame_stacking is None:
            obs = self.obs[indices]
        else:
            batch_size: int = indices[0].shape[0]
            obs = np.empty((batch_size, frame_stacking, *self.obs.shape[2:]),
                           dtype=self.obs.dtype)
            for i in range(frame_stacking):
                if i:
                    dones = ~self.masks[indices]
                    obs[dones, :i] = np.expand_dims(
                        self.obs[indices[0][dones], indices[1][dones]], axis=1)
                    indices = self.next_indices(indices)
                obs[:, i] = self.obs[indices]
        return self.decompress_fn(obs), indices

    def get_data(
        self,
        indices: Tuple[np.ndarray, np.ndarray],
        frame_stacking: Optional[int] = None,
    ) -> Tuple[np.ndarray, ...]:
        next_indices = self.next_indices(indices, self.n_steps)

        obs, indices = self.get_obs(indices, frame_stacking)
        next_obs, next_indices = self.get_obs(next_indices, frame_stacking)

        return (obs, self.actions[indices], self.rewards[indices],
                self.multi_masks[indices], next_obs)

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
               ex_steps: int = 0,
               frame_stacking: int = None):
        frame_steps = 0 if frame_stacking is None else (frame_stacking - 1)
        indices = self.sample_indices(rng, batch_size, ex_steps + frame_steps)
        return (*self.get_data(indices, frame_stacking), indices)


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
                 compress_fn: TransformFunction = None,
                 decompress_fn: TransformFunction = None,
                 alpha: float = PRIORITY_ALPHA,
                 beta: float = PRIORITY_BETA):
        if isinstance(obs_shape, int):
            obs_shape = obs_shape,
        super().__init__(num_envs, cap, obs_shape, obs_dtype, reward_scaling,
                         n_steps, discount, compress_fn, decompress_fn)
        self.priorities = tuple(SumTree(cap) for _ in range(num_envs))
        self.max_priority = 2.

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
             err: float = None,
             compress: bool = True) -> int:
        if err is not None:
            priority = self._get_priority(err)
            self.max_priority = max(self.max_priority, priority)
        else:
            priority = self.max_priority
        idx = super().push(env_id, obs, action, reward, done, compress)
        self.priorities[env_id][idx] = priority
        return idx

    def update_priorities(self, indices: Tuple[Iterable[int], Iterable[int]],
                          errs: np.ndarray):
        priorities = self._get_priority(errs)
        max_priority = priorities.max()
        if max_priority < self.max_priority:
            self.max_priority *= MAX_PRIORITY_ADAPT_SPEED
            self.max_priority += (1 - MAX_PRIORITY_ADAPT_SPEED) * max_priority
        else:
            self.max_priority = max_priority
        for env_id, i, p in zip(indices[0], indices[1], priorities):
            self.priorities[env_id][i] = p

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
               ex_steps: int = 0,
               frame_stacking: int = None):
        frame_steps = 0 if frame_stacking is None else (frame_stacking - 1)
        weights, indices = self.sample_indices(rng, batch_size,
                                               ex_steps + frame_steps)
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
                 compress_fn: TransformFunction = None,
                 decompress_fn: TransformFunction = None,
                 alpha: float = PRIORITY_ALPHA,
                 beta: float = PRIORITY_BETA):
        super().__init__(num_envs, cap, obs_shape, obs_dtype, reward_scaling,
                         n_steps, discount, compress_fn, decompress_fn, alpha,
                         beta)
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
             mem: List[np.ndarray] = None,
             compress: bool = True) -> int:
        if mem is None:
            raise TypeError
        idx = super().push(env_id, obs, action, reward, done, err, compress)
        for i, m in enumerate(mem):
            self.mem[i][env_id, idx] = m
        return idx

    def update_memory(self, indices: np.ndarray, mem: List[np.ndarray]):
        for i, m in enumerate(mem):
            self.mem[i][indices] = m

    def get_data(self,
                 indices: Tuple[np.ndarray, np.ndarray],
                 get_mem: bool = True,
                 frame_stacking: Optional[int] = None):
        next_indices = self.next_indices(indices, self.n_steps)

        obs, indices = self.get_obs(indices, frame_stacking)
        next_obs, next_indices = self.get_obs(next_indices, frame_stacking)

        if get_mem:
            mem = [m[indices] for m in self.mem]
            next_mem = [m[next_indices] for m in self.mem]
            return (mem, obs, self.actions[indices], self.masks[indices],
                    self.rewards[indices], self.multi_masks[indices], next_mem,
                    next_obs, self.masks[next_indices])
        else:
            return (obs, self.actions[indices], self.masks[indices],
                    self.rewards[indices], self.multi_masks[indices], next_obs,
                    self.masks[next_indices])

    def sample(self,
               rng: random.Generator,
               batch_size: int,
               ex_steps: int = 0,
               frame_stacking: int = None,
               get_mem: bool = True):
        frame_steps = 0 if frame_stacking is None else (frame_stacking - 1)
        weights, indices = self.sample_indices(rng, batch_size,
                                               ex_steps + frame_steps)
        return (*self.get_data(indices, get_mem, frame_stacking), weights,
                indices)
