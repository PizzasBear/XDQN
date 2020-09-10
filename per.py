from typing import Tuple, Union
import numpy as np
from numpy import random

ALPHA: float = 0.6
BETA: float = 0.4
REWARD_SCALING: float = 1.


class SumTree:
    tree: np.ndarray
    cap: int

    def __init__(self, cap):
        self.tree = np.zeros(2 * cap - 1)
        self.cap = cap

    def total(self):
        return self.tree[0]

    def __getitem__(self, item):
        return self.tree[item + self.cap - 1]

    def __setitem__(self, key, value):
        tree_idx = key + self.cap - 1
        change = value - self.tree[tree_idx]
        self.tree[tree_idx] = value
        while tree_idx:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, s, idx=0) -> Tuple[int, float]:
        left = 2 * idx + 1
        right = left + 1

        if len(self.tree) <= left:
            return idx - self.cap + 1, self.tree[idx]

        if s <= self.tree[left]:
            return self.get(s, left)
        else:
            return self.get(s - self.tree[left], right)


class ReplayBuffer:
    __slots__ = ('obs', 'actions', 'rewards', 'masks', 'next_indices',
                 'prev_indices', 'reward_scaling', 'cap', '_len',
                 '_current_index')

    # Buffers
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
    next_indices: np.ndarray
    prev_indices: np.ndarray
    # Hyper Params
    reward_scaling: float
    # Buffer info
    cap: int
    _len: int
    _current_index: int

    def __init__(self,
                 cap: int,
                 obs_shape: Union[int, Tuple[int, ...]],
                 obs_dtype: np.dtype = np.float32,
                 reward_scaling: float = REWARD_SCALING):
        if isinstance(obs_shape, int):
            obs_shape = obs_shape,
        self.prev_indices = np.zeros(cap, dtype=np.int32)
        self.obs = np.zeros((cap, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros(cap, dtype=np.int32)
        self.rewards = np.zeros(cap, dtype=np.float32)
        self.masks = np.zeros(cap, dtype=np.bool)
        self.next_indices = np.zeros(cap, dtype=np.int32)

        self.cap = cap
        self._len = 0
        self._current_index = 0
        self.reward_scaling = reward_scaling

    def __len__(self) -> int:
        return self._len

    def push(self, prev_idx: int, obs: np.ndarray, action: int, reward: float,
             done: bool) -> int:
        if self._len < self.cap:
            self._len += 1
        idx = self._current_index
        self._current_index += 1
        self._current_index %= self.cap

        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward / self.reward_scaling
        self.masks[idx] = not done

        self.next_indices[prev_idx] = idx
        self.prev_indices[idx] = prev_idx
        return idx

    def sample(self,
               rng: random.Generator,
               batch_size: int,
               discount: float,
               steps: int = 1) -> Tuple[np.ndarray, ...]:
        indices = rng.integers(0, self._len - steps, batch_size)
        rewards = self.rewards[indices]
        masks = self.masks[indices]
        i = indices
        for j in range(steps - 1):
            rewards += masks * (discount**j) * self.rewards[i]
            masks = np.all([masks, self.masks[i]], axis=0)
            i = self.next_indices[i]

        return (self.obs[indices], self.actions[indices], rewards, masks,
                self.obs[i], indices)


class PrioritisedReplayBuffer(ReplayBuffer):
    __slots__ = 'priorities', 'alpha', 'beta'

    # Buffers
    priorities: SumTree
    # Hyper Params
    alpha: float
    beta: float

    def __init__(self,
                 cap: int,
                 obs_shape: Union[int, Tuple[int, ...]],
                 obs_dtype: np.dtype = np.float32,
                 alpha: float = ALPHA,
                 beta: float = BETA,
                 reward_scaling: float = REWARD_SCALING):
        if isinstance(obs_shape, int):
            obs_shape = obs_shape,
        super().__init__(cap, obs_shape, obs_dtype, reward_scaling)
        self.priorities = SumTree(cap)

        self.alpha = alpha
        self.beta = beta

    def _get_priority(self, err, epsilon=0.01):
        return (np.abs(err) + epsilon)**self.alpha

    def push(self,
             prev_idx: int,
             obs: np.ndarray,
             action: int,
             reward: float,
             done: bool,
             err: float = None) -> int:
        if err is None:
            raise RuntimeError
        idx = super().push(prev_idx, obs, action, reward, done)
        self.priorities[idx] = self._get_priority(err)
        return idx

    def update_priorities(self, indices, errs):
        for i, err in zip(indices, errs):
            self.priorities[i] = self._get_priority(err)

    def sample(self,
               rng: random.Generator,
               batch_size: int,
               discount: float,
               steps: int = 1) -> Tuple[np.ndarray, ...]:
        indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.int32)
        for i in range(batch_size):
            idx, p = self.priorities.get(
                rng.uniform(0., self.priorities.total()))
            if self._current_index - steps <= idx < self._current_index:
                idx = self._current_index - steps - 1
            weights[i] = (self._len * p / self.priorities.total())**-self.beta
            indices[i] = idx
        weights /= weights.max(initial=0.01)

        rewards = self.rewards[indices]
        masks = self.masks[indices]
        i = indices
        for j in range(steps - 1):
            rewards += masks * (discount**j) * self.rewards[i]
            masks = np.all([masks, self.masks[i]], axis=0)
            i = self.next_indices[i]

        return (self.obs[indices], self.actions[indices], rewards, masks,
                self.obs[i], weights, indices)
