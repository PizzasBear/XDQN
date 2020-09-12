import dqn
import distrib
import traceback
from typing import Dict
from datetime import datetime
from itertools import count
import numpy as np
from numpy import random
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from multiprocessing.connection import Connection
import gym

NUM_ACTORS: int = 10
ACTOR_UPDATE_INTERVAL: int = 500
SAVE_INTERVAL: int = 1000
REPLAY_BUFFER_UPDATE_INTERVAL: int = 500
IN_FEATURES: int = 8
NUM_ACTIONS: int = 4
ENV_ID: str = 'LunarLander-v2'
LEARNER_DEVICE: str = 'cuda'
ACTOR_DEVICE: str = 'cpu'
LOG_DIR: str = 'lunar-lander/'
MODEL_FILE: str = 'lunar-lander.pt'


def make_env():
    return gym.make(ENV_ID)


class ActorBuffer:
    __slots__ = ('prev_idx', 'obs', 'actions', 'qs', 'rewards', 'dones')

    prev_idx: int
    obs: np.ndarray
    actions: np.ndarray
    qs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    def __init__(self, prev_idx: int, obs: np.ndarray, actions: np.ndarray,
                 qs: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        self.prev_idx = prev_idx
        self.obs = obs
        self.actions = actions
        self.qs = qs
        self.rewards = rewards
        self.dones = dones


def make_net(in_features: int, num_actions: int):
    net = dqn.DuelingQNet(
        dqn.MLP([64], in_features),
        dqn.MLP([48], 64, out_features=1),
        dqn.MLP([48], 64, out_features=num_actions),
    )
    return net


def actor_proc(conn: Connection):
    try:
        actor = dqn.Actor(make_net(IN_FEATURES, NUM_ACTIONS)).to(ACTOR_DEVICE)
        actor.load_state_dict(conn.recv())
        rng: random.Generator = random.default_rng()
        env: gym.Env = make_env()
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
        can_send_fitness = False
        fitness = 0.
        for t in count():
            i = t % REPLAY_BUFFER_UPDATE_INTERVAL
            # Play
            action, q = actor.act(obs, t, rng, ACTOR_DEVICE)
            next_obs, reward, done, _ = env.step(action)

            # Send
            if not i and can_send:
                buffer.obs[-1] = next_obs
                conn.send(('buffer', buffer))
                while True:
                    key, value = conn.recv()
                    if key == 'prev_idx':
                        buffer.prev_idx = value
                        break
                    elif key == 'net':
                        actor.load_state_dict(value)
                    elif key == 'send_fitness':
                        can_send_fitness = True
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
                if can_send_fitness:
                    conn.send(('fitness', fitness))  # send fitness
                next_obs = env.reset()
                fitness = 0.
            obs = next_obs

            if conn.poll(0):
                key, value = conn.recv()
                if key == 'net':
                    actor.load_state_dict(value)
                elif key == 'send_fitness':
                    can_send_fitness = True
                else:
                    raise RuntimeError
    except Exception as e:
        conn.send(('except', (e, traceback.format_exc())))


def update_replay_buffer(agent: dqn.Agent, buff: ActorBuffer, device: dqn.Device = LEARNER_DEVICE) -> int:
    prev_idx = buff.prev_idx
    for i in range(REPLAY_BUFFER_UPDATE_INTERVAL):
        prev_idx = agent.store(prev_idx,
                               buff.qs[i],
                               buff.obs[i],
                               buff.actions[i],
                               buff.rewards[i],
                               buff.dones[i],
                               buff.obs[i + 1],
                               device=device)
    return prev_idx


def state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    for k, v in state_dict.items():
        state_dict[k] = v.cpu()
    return state_dict


def learner_proc(conn: distrib.MultiConnection):
    agent = dqn.Agent(
        make_net(IN_FEATURES, NUM_ACTIONS),
        make_net(IN_FEATURES, NUM_ACTIONS),
        IN_FEATURES,
    ).to(LEARNER_DEVICE)
    conn.send(state_dict_to_cpu(agent.state_dict()))

    log_dir: str = 'data/logs/' + LOG_DIR + datetime.now().strftime(
        '%Y%m%d-%H%M%S')
    writer: SummaryWriter = SummaryWriter(log_dir)
    rng: random.Generator = random.default_rng()

    # collect experience 'till the agent can learn
    while not agent.can_learn():
        idx, (key, value) = conn.recv()
        if key == 'buffer':
            conn.send(('prev_idx', update_replay_buffer(agent, value)), idx)
        elif key == 'except':
            print(value[1])
            raise value[0]
    conn.send(('send_fitness', None))
    # start the learning loop
    for t in count(1):
        agent.learn(t, writer, rng, LEARNER_DEVICE)
        # update the actor
        if not t % ACTOR_UPDATE_INTERVAL:
            conn.send(('net', state_dict_to_cpu(agent.state_dict())))
        if not t % SAVE_INTERVAL:
            torch.save(agent.state_dict(), 'models/' + MODEL_FILE)
        # collect experience
        if conn.poll(0):
            idx, (key, value) = conn.recv()
            if key == 'buffer':
                conn.send(('prev_idx', update_replay_buffer(agent, value)),
                          idx)
            elif key == 'fitness':
                writer.add_scalar('Fitness', value, t)
            elif key == 'except':
                print(value[1])
                raise value[0]


def main():
    actor_conns, learner_conns = distrib.MultiPipe(NUM_ACTORS)
    actor_handles = [
        mp.Process(target=actor_proc, args=(conn, ), daemon=True)
        for conn in actor_conns
    ]
    for actor_handle in actor_handles:
        actor_handle.start()
    learner_proc(distrib.MultiConnection(learner_conns))


if __name__ == '__main__':
    main()
