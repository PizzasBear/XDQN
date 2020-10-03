import time
from itertools import cycle
import numpy as np
import gym
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import torch
from typing import List
from xdqn import algo
from xdqn.buffers import REWARD_SCALING, FrameStack
from xdqn.train import SAVE_INTERVAL, FRAME_STACKING, MODEL_FILE, common_pre_proc, pre_proc, frame_stack_pre_proc, make_net, make_env

Q_UPDATE_INTERVAL: int = 3
LOAD_INTERVAL: int = 2 * SAVE_INTERVAL
AVG_FITNESS_SPEED: float = 0.1
ADDITIONAL_SPACE: float = 0.05
FIRE_INTERVAL: int = 200
DISCOUNT: float = 1.
ZOOM_IN: float = 0.997

env = make_env()

state_info: gym.spaces.Box = env.observation_space
action_info: gym.spaces.Discrete = env.action_space
env.render()
# noinspection PyTypeChecker
net = make_net(env.action_space.n)
agent = algo.Agent(
    net,
    None,
    1,
    (),
    net.mem_size(),
    num_quantiles=algo.NUM_QUANTILES,
    frame_stacking=FRAME_STACKING,
    train=False,
)
mem = agent.get_init_mem(1)
obs: np.ndarray = pre_proc(common_pre_proc(env.reset()))
frame_stack = FrameStack(obs, FRAME_STACKING)

fig: plt.Figure = plt.figure()
ax1, ax2 = fig.subplots(ncols=2)

min_q, max_q = -20., 20.
min_advantage, max_advantage = -1., 1.

first_episode: bool = True
fitness: float = 0.
avg_fitness: float = 0.
discounted_fitness: float = 0.
avg_discounted_fitness: float = 0.
q_bars: List[Rectangle] = ax1.bar(range(env.action_space.n + 2),
                                  [0.0 for _ in range(env.action_space.n + 2)])
q_bars[0].set_color('tab:green')
q_bars[1].set_color('tab:red')
advantage_bars: List[Rectangle] = ax2.bar(range(
    env.action_space.n), [0.0 for _ in range(env.action_space.n)])

fig.show()

discount: float = 1.
for t in cycle(range(LOAD_INTERVAL)):
    if not t:
        print('reloading...')
        agent.net.load_state_dict(
            torch.load('models/' + MODEL_FILE, map_location='cpu'))
    if env.render() is False:
        break
    (action, ), (qs, ), mem = agent.act(
        mem,
        frame_stack_pre_proc(np.expand_dims(frame_stack.frames, 0)),
        None,
        get_q=True)
    if not t % FIRE_INTERVAL:
        action = 1
    advantages = qs - qs.mean()
    obs, reward, done, _ = env.step(action)
    fitness += reward
    discounted_fitness += discount * reward
    if not t % Q_UPDATE_INTERVAL:
        advantages *= REWARD_SCALING
        qs *= REWARD_SCALING
        for i, (q, q_bar, advantage, advantage_bar) in enumerate(
                zip(qs, q_bars[2:], advantages, advantage_bars)):
            q_bar.set_height(q)
            q_bar.set_color('tab:blue')
            advantage_bar.set_height(advantage)
            advantage_bar.set_color('tab:blue')
        q_bars[1].set_height(qs.mean())
        if first_episode:
            avg_fitness = fitness
        avg_curr_value = (avg_discounted_fitness - discounted_fitness
                          ) / discount if not first_episode else 0
        q_bars[0].set_height(avg_curr_value)
        q_bars[action + 2].set_color('tab:orange')
        advantage_bars[action].set_color('tab:orange')
        # 2 4 6
        avg_q = 0.5 * (max_q + min_q)
        min_q = qs.min(initial=min((min_q - avg_q) * ZOOM_IN +
                                   avg_q, avg_curr_value))
        max_q = qs.max(initial=max((max_q - avg_q) * ZOOM_IN +
                                   avg_q, avg_curr_value))
        avg_advantage = 0.5 * (min_advantage + max_advantage)
        min_advantage = advantages.min(
            initial=(min_advantage - avg_advantage) * ZOOM_IN + avg_advantage)
        max_advantage = advantages.max(
            initial=(max_advantage - avg_advantage) * ZOOM_IN + avg_advantage)
        ax1.set_ylim(min_q + (min_q - max_q) * ADDITIONAL_SPACE,
                     max_q + (max_q - min_q) * ADDITIONAL_SPACE)
        ax2.set_ylim(
            min_advantage + (min_advantage - max_advantage) * ADDITIONAL_SPACE,
            max_advantage + (max_advantage - min_advantage) * ADDITIONAL_SPACE)
        fig.canvas.draw()

    if done:
        mem = agent.get_init_mem(1)
        obs = env.reset()
        if first_episode:
            avg_fitness = fitness
            avg_discounted_fitness = discounted_fitness
            first_episode = False
        else:
            avg_fitness *= 1 - AVG_FITNESS_SPEED
            avg_fitness += fitness * AVG_FITNESS_SPEED
            avg_discounted_fitness *= 1 - AVG_FITNESS_SPEED
            avg_discounted_fitness += discounted_fitness * AVG_FITNESS_SPEED
        print(f'fitness: {fitness} ~ {avg_fitness}')
        fitness = 0.
        discounted_fitness = 0.
        discount = 1 / DISCOUNT
    obs = pre_proc(common_pre_proc(obs))
    frame_stack.update(done, obs)
    time.sleep(1 / 60)
    discount *= DISCOUNT
