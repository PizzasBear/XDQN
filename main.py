from datetime import datetime

import dqn
import distrib
import multiprocessing as mp
import numpy as np
from numpy import random
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from itertools import count, cycle
from collections import namedtuple
import gym

NUM_ACTORS: int = 10
ACTOR_UPDATE_INTERVAL: int = 500
REPLAY_BUFFER_UPDATE_INTERVAL: int = 500
IN_FEATURES: int = 8
NUM_ACTIONS: int = 4
ENV_ID: str = 'LunarLander-v2'
LOG_FOLDER: str = 'lunar-lander'
DEVICE: str = 'cuda'

ActorBuffer = namedtuple(
    'ActorBuffer', ('prev_idx', 'obs', 'actions', 'qs', 'rewards', 'dones'))


def create_net(in_features: int, num_actions: int):
    return dqn.DuelingQNet(
        dqn.MLP([128], in_features),
        dqn.MLP([128], in_features, out_features=1),
        dqn.MLP([128], in_features, out_features=num_actions),
    )


def actor_proc(conn: mp.connection.Connection):
    actor = dqn.Actor(create_net(IN_FEATURES, NUM_ACTIONS)).to(DEVICE)
    actor.load_state_dict(conn.recv())

    rng: random.Generator = random.default_rng()
    env: gym.Env = gym.make(ENV_ID)
    obs: np.ndarray = env.reset()
    buffer = ActorBuffer(
        0,
        np.zeros((REPLAY_BUFFER_UPDATE_INTERVAL + 1, IN_FEATURES),
                 dtype=np.float32),
        np.zeros(REPLAY_BUFFER_UPDATE_INTERVAL, dtype=np.int32),
        np.zeros(REPLAY_BUFFER_UPDATE_INTERVAL, dtype=np.float32),
        np.zeros(REPLAY_BUFFER_UPDATE_INTERVAL, dtype=np.float32),
        np.zeros(REPLAY_BUFFER_UPDATE_INTERVAL, dtype=np.bool),
    )

    can_send = False
    fitness = 0.
    for i in cycle(range(REPLAY_BUFFER_UPDATE_INTERVAL)):
        # Play
        action, q = actor.act(obs, rng, DEVICE)
        obs, reward, done, next_obs = env.step(action)

        # Send
        if not i and can_send:
            buffer.obs[-1] = next_obs
            conn.send(('buffer', buffer))
            key, value = conn.recv()
            if key == 'prev_idx':
                buffer.prev_idx = value
            else:
                raise RuntimeError

        # Save to buffer
        buffer.obs[i] = obs
        buffer.actions[i] = action
        buffer.qs[i] = q
        buffer.rewards[i] = reward
        buffer.dones[i] = done

        # Update obs and fitness
        can_send = True
        fitness += reward
        if done:
            conn.send(('fitness', fitness))  # send fitness
            next_obs = env.reset()
            fitness = 0.
        obs = next_obs

        if conn.poll(0):
            key, value = conn.recv()
            if key == 'net':
                actor.load_state_dict(value)


def update_replay_buffer(agent: dqn.Agent, buff: ActorBuffer) -> int:
    prev_idx = buff.prev_idx
    for i in range(REPLAY_BUFFER_UPDATE_INTERVAL):
        prev_idx = agent.store(prev_idx,
                               buff.qs[i],
                               buff.obs[i],
                               buff.actions[i],
                               buff.rewards[i],
                               buff.dones[i],
                               buff.obs[i + 1],
                               device=DEVICE)
    return prev_idx


def learner_proc(conn: distrib.MultiConnection):
    agent = dqn.Agent(
        create_net(IN_FEATURES, NUM_ACTIONS),
        create_net(IN_FEATURES, NUM_ACTIONS),
        IN_FEATURES,
    ).to(DEVICE)
    conn.send(agent.state_dict())

    log_dir: str = 'data/logs/' + LOG_FOLDER + datetime.now().strftime(
        '%Y%m%d-%H%M%S')
    writer: SummaryWriter = SummaryWriter(log_dir)
    rng: random.Generator = random.default_rng()

    # collect experience 'till the agent can learn
    while not agent.can_learn():
        if conn.poll_any():
            idx, (key, value) = conn.recv()
            if key == 'buffer':
                conn.send(('prev_idx', update_replay_buffer(agent, value)),
                          idx)
    # start the learning loop
    for t in count(1):
        agent.learn(t, writer, rng, DEVICE)
        # update the actor
        if not t % ACTOR_UPDATE_INTERVAL:
            conn.send(('net', agent.state_dict()))
        # collect experience
        if conn.poll_any():
            idx, (key, value) = conn.recv()
            if key == 'buffer':
                conn.send(('prev_idx', update_replay_buffer(agent, value)),
                          idx)
            elif key == 'fitness':
                writer.add_scalar('Fitness', value, t)


def main():
    actor_conns, learner_conns = distrib.MultiPipe(NUM_ACTORS)
    actor_handles = [
        mp.Process(target=actor_proc, args=(conn, )) for conn in actor_conns
    ]
    for actor_handle in actor_handles:
        actor_handle.start()
    learner_proc(distrib.MultiConnection(learner_conns))


if __name__ == '__main__':
    main()
