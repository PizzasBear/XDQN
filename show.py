import time
from itertools import cycle
import numpy as np
import gym
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import torch
from typing import List
import per
from train import SAVE_INTERVAL, MODEL_FILE, IN_FEATURES, NUM_ACTIONS, make_net, make_env
import xdqn

Q_UPDATE_INTERVAL: int = 6
LOAD_INTERVAL: int = SAVE_INTERVAL
AVG_FITNESS_SPEED: float = 0.1
ADDITIONAL_SPACE: float = 0.05

env = make_env()

state_info: gym.spaces.Box = env.observation_space
action_info: gym.spaces.Discrete = env.action_space
env.render()
# noinspection PyTypeChecker
actor = make_net(IN_FEATURES, NUM_ACTIONS)
actor = xdqn.Actor(actor, 0.)
obs: np.ndarray = env.reset()

fig: plt.Figure = plt.figure()
ax1, ax2 = fig.subplots(ncols=2)

min_q, max_q = -20., 20.
min_advantage, max_advantage = -1., 1.

first_fitness: bool = True
fitness: float = 0.
avg_fitness: float = 0.
q_bars: List[Rectangle] = ax1.bar(range(NUM_ACTIONS + 2),
                                  [0.0 for _ in range(NUM_ACTIONS + 2)])
q_bars[0].set_color('tab:green')
q_bars[1].set_color('tab:red')
advantage_bars: List[Rectangle] = ax2.bar(range(NUM_ACTIONS),
                                          [0.0 for _ in range(NUM_ACTIONS)])

fig.show()

for t in cycle(range(LOAD_INTERVAL)):
    if not t:
        actor.load_state_dict(torch.load('models/' + MODEL_FILE))
    if env.render() is False:
        break
    action, qs = actor.act(obs, None, None, 'cpu', True)
    advantages = qs - qs.mean()
    obs, reward, done, _ = env.step(action)
    fitness += reward
    if not t % Q_UPDATE_INTERVAL:
        advantages *= per.REWARD_SCALING
        qs *= per.REWARD_SCALING
        for i, (q, q_bar, advantage, advantage_bar) in enumerate(
                zip(qs, q_bars[2:], advantages, advantage_bars)):
            q_bar.set_height(q)
            q_bar.set_color('tab:blue')
            advantage_bar.set_height(advantage)
            advantage_bar.set_color('tab:blue')
        q_bars[1].set_height(qs.mean())
        q_bars[0].set_height(avg_fitness)
        q_bars[action + 1].set_color('tab:orange')
        advantage_bars[action].set_color('tab:orange')
        min_q = qs.min(initial=min(min_q, avg_fitness))
        max_q = qs.max(initial=max(max_q, avg_fitness))
        min_advantage = advantages.min(initial=min_advantage)
        max_advantage = advantages.max(initial=max_advantage)
        ax1.set_ylim(min_q + (min_q - max_q) * ADDITIONAL_SPACE,
                     max_q + (max_q - min_q) * ADDITIONAL_SPACE)
        ax2.set_ylim(
            min_advantage + (min_advantage - max_advantage) * ADDITIONAL_SPACE,
            max_advantage + (max_advantage - min_advantage) * ADDITIONAL_SPACE)
        fig.canvas.draw()

    if done:
        print(f'fitness: {fitness}')
        obs = env.reset()
        if first_fitness:
            avg_fitness = fitness
            first_fitness = False
        else:
            avg_fitness *= 1 - AVG_FITNESS_SPEED
            avg_fitness += fitness * AVG_FITNESS_SPEED
        fitness = 0.
    time.sleep(1 / 120)
