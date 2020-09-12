import time
from itertools import cycle

import numpy as np
import gym
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import torch

import per
from main import MODEL_FILE, IN_FEATURES, NUM_ACTIONS, make_net, make_env
import dqn

env = make_env()

state_info: gym.spaces.Box = env.observation_space
action_info: gym.spaces.Discrete = env.action_space
env.render()
# noinspection PyTypeChecker
actor = make_net(IN_FEATURES, NUM_ACTIONS)
actor = dqn.Actor(actor, 0.)
actor.load_state_dict(torch.load('models/' + MODEL_FILE))
obs: np.ndarray = env.reset()

fig: plt.Figure = plt.figure()
ax1, ax2 = fig.subplots(ncols=2)

min_q, max_q = -20., 20.
min_advantage, max_advantage = -1., 1.

fitness: float = 0.
q_bars = ax1.bar(range(NUM_ACTIONS + 1), [0.0 for _ in range(NUM_ACTIONS + 1)])
advantage_bars = ax2.bar(range(NUM_ACTIONS), [0.0 for _ in range(NUM_ACTIONS)])

fig.show()

for t in cycle(range(6)):
    if env.render() is False:
        break
    action, qs = actor.act(obs, None, None, 'cpu', True)
    advantages = qs - qs.mean()
    obs, reward, done, _ = env.step(action)
    fitness += reward
    if not t:
        for i, (q, q_bar, advantage, advantage_bar) in enumerate(zip(qs, q_bars[1:], advantages, advantage_bars)):
            q_bar.set_height(q * per.REWARD_SCALING)
            advantage_bar.set_height(advantage * per.REWARD_SCALING)
        q_bars[0].set_height(qs.mean() * per.REWARD_SCALING)
        min_q, max_q = qs.min(initial=min_q), qs.max(initial=max_q)
        min_advantage = advantages.min(initial=min_advantage)
        max_advantage = advantages.max(initial=max_advantage)
        ax1.set_ylim(min_q, max_q)
        ax2.set_ylim(min_advantage, max_advantage)
        fig.canvas.draw()

    if done:
        print(f'fitness: {fitness}')
        obs = env.reset()
        fitness = 0.
    time.sleep(1 / 120)
