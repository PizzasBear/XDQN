from xdqn import nets, algo, distrib, buffers
from typing import Optional, Union, Tuple
from datetime import datetime
from itertools import count
import numpy as np
from numpy import random
import torch
from torch.utils.tensorboard import SummaryWriter
import gym
import gym.wrappers
from xdqn.consts import *

# Environment
ENV_ID: str = 'BreakoutNoFrameskip-v4'
FRAME_STACKING: Optional[int] = 4
COMPRESSED_OBS_SHAPE: Union[int, Tuple[int, ...]] = (84, 84)
COMPRESSED_OBS_DTYPE: np.dtype = np.uint8
PROC_FEATURES_SHAPE: Tuple[int, ...] = (FRAME_STACKING, 84, 84)
NUM_ENVS: int = 10
FIRE_INTERVAL: int = 200

# Storage info
SAVE_INTERVAL: int = 1500
DEVICE: str = 'cuda'
LOG_DIR: str = 'breakout/'
MODEL_FILE: str = 'breakout.pt'
CONTINUE: bool = False


def make_env():
    return gym.wrappers.AtariPreprocessing(gym.make(ENV_ID), terminal_on_life_loss=True)


def make_net(num_actions: int):
    feature_net = nets.ConvNet([(32, 8, 4), (64, 4, 2), (64, 3, 1)], 4, True)
    feature_net_out_size = feature_net(torch.zeros(
        1, *PROC_FEATURES_SHAPE)).size()[-1]
    net = nets.QuantileDuelingQNet(
        feature_net=feature_net,
        memory_net=nets.LSTM([512], feature_net_out_size),
        value_net=nets.MLP([512], 512, out_features=algo.NUM_QUANTILES),
        advantage_net=nets.MLP([512],
                               512,
                               out_features=num_actions * algo.NUM_QUANTILES),
    )
    return net


def common_pre_proc(obs: np.ndarray) -> np.ndarray:
    return obs


def pre_proc(obs: np.ndarray) -> np.ndarray:
    return obs


def frame_stack_pre_proc(obs: np.ndarray) -> np.ndarray:
    return obs.astype(np.float32) / 256.


def compress(obs: np.ndarray) -> np.ndarray:
    return obs


def decompress(obs: np.ndarray) -> np.ndarray:
    return obs.astype(np.float32) / 256.


def main():
    log_dir: str = 'data/logs/' + LOG_DIR + datetime.now().strftime(
        '%Y%m%d-%H%M%S')
    writer: SummaryWriter = SummaryWriter(log_dir)
    rng: random.Generator = random.default_rng()

    envs = distrib.MPVecEnv(make_env, NUM_ENVS)

    epsilon = BASE_EPSILON**(1 +
                             EPSILON_ALPHA * np.flip(np.arange(NUM_ENVS), 0) /
                             (NUM_ENVS - 2))
    epsilon[0] = 0

    net = make_net(envs.action_space.n)
    if CONTINUE:
        net.load_state_dict(
            torch.load('models/' + MODEL_FILE, map_location=DEVICE))
    agent = algo.Agent(
        net,
        make_net(envs.action_space.n),
        NUM_ENVS,
        COMPRESSED_OBS_SHAPE,
        net.mem_size(),
        obs_dtype=COMPRESSED_OBS_DTYPE,
        num_quantiles=algo.NUM_QUANTILES,
        compress_fn=compress,
        decompress_fn=decompress,
        frame_stacking=FRAME_STACKING,
    ).to(DEVICE)

    fitness = np.zeros(NUM_ENVS)

    obs = common_pre_proc(envs.reset())
    frame_stack = buffers.VecFrameStack(obs, FRAME_STACKING)
    mem = agent.get_init_mem(NUM_ENVS)
    for t in count():
        if not t % SAVE_INTERVAL:
            torch.save(agent.net_state_dict(), 'models/' + MODEL_FILE)
        t *= NUM_ENVS
        actions, next_mem = agent.act(mem,
                                      frame_stack_pre_proc(frame_stack.frames),
                                      rng,
                                      device=DEVICE,
                                      epsilon=epsilon)
        if not t % FIRE_INTERVAL:
            actions.fill(1)
        envs.step_async(actions)
        if agent.can_learn():
            agent.learn(t, writer, rng, DEVICE)
        next_obs, rewards, dones = envs.await_step()
        next_obs = common_pre_proc(next_obs)
        for i in range(NUM_ENVS):
            agent.store(i, [m[i] for m in mem], obs[i], actions[i], rewards[i],
                        dones[i])

        # Update obs and fitness
        mem = algo.mask_mem(torch.tensor(dones, device=DEVICE),
                            agent.get_init_mem(NUM_ENVS), next_mem)
        fitness += rewards
        for i, done in enumerate(dones):
            if done:
                if i == 0:
                    writer.add_scalar('Eval Fitness', fitness[0], t + i)
                writer.add_scalar('Fitness', fitness[i], t + i)
                fitness[i] = 0.
        frame_stack.update(dones, pre_proc(next_obs))
        obs = next_obs


if __name__ == '__main__':
    main()
