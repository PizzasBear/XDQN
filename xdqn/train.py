from xdqn import nets, algo, distrib, buffers
import traceback
from typing import Dict, Optional
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
ACTOR_UPDATE_INTERVAL: int = 293
SAVE_INTERVAL: int = 1000
REPLAY_BUFFER_UPDATE_INTERVAL: int = 223
IN_FEATURES: int = 8
NUM_ACTIONS: int = 4
ENV_ID: str = 'LunarLander-v2'
LEARNER_DEVICE: str = 'cuda'
ACTOR_DEVICE: str = 'cpu'
LOG_DIR: str = 'lunar-lander/'
MODEL_FILE: str = 'lunar-lander.pt'
BASE_EPSILON: float = 0.4
EPSILON_ALPHA: float = 7.
CONTINUE: bool = False


def make_env():
    return gym.make(ENV_ID)


def make_net(in_features: int, num_actions: int):
    net = nets.QuantileDuelingQNet(
        feature_net=nets.MLP([64], in_features),
        memory_net=nets.LSTM([64], 64),
        value_net=nets.MLP([48], 64, out_features=algo.NUM_QUANTILES),
        advantage_net=nets.MLP([48],
                               64,
                               out_features=num_actions * algo.NUM_QUANTILES),
    )
    return net


def actor_proc(conn: Connection, epsilon: Optional[float] = None):
    try:
        actor = algo.Actor(make_net(IN_FEATURES, NUM_ACTIONS)).to(ACTOR_DEVICE)
        actor.load_state_dict(conn.recv())
        rng: random.Generator = random.default_rng()
        env: gym.Env = make_env()
        obs: np.ndarray = env.reset()
        mem = actor.init_mem(1, 'cpu')

        buffer = buffers.RecurrentActorBuffer(
            REPLAY_BUFFER_UPDATE_INTERVAL,
            IN_FEATURES,
            actor.mem_size(),
        )

        can_send_fitness = False
        fitness = 0.
        while True:
            # Play
            action, mem = actor.act(mem,
                                    obs,
                                    rng,
                                    ACTOR_DEVICE,
                                    epsilon=epsilon)
            next_obs, reward, done, _ = env.step(action)

            # Send
            if buffer.can_send():
                conn.send(('buffer', buffer))
                buffer.clear()

            buffer.push(obs, action, reward, done, mem)

            # Update obs and fitness
            fitness += reward
            if done:
                mem = actor.init_mem(1, 'cpu')
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


def state_dict_to_cpu(
        state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    for k, v in state_dict.items():
        state_dict[k] = v.cpu()
    return state_dict


def learner_proc(conn: distrib.MultiConnection):
    net = make_net(IN_FEATURES, NUM_ACTIONS)
    if CONTINUE:
        net.load_state_dict(torch.load('models/' + MODEL_FILE))
    agent = algo.Agent(
        net,
        make_net(IN_FEATURES, NUM_ACTIONS),
        NUM_ACTORS,
        IN_FEATURES,
        net.mem_size(),
        num_quantiles=algo.NUM_QUANTILES,
    ).to(LEARNER_DEVICE)
    conn.send(state_dict_to_cpu(agent.net_state_dict()))

    log_dir: str = 'data/logs/' + LOG_DIR + datetime.now().strftime(
        '%Y%m%d-%H%M%S')
    writer: SummaryWriter = SummaryWriter(log_dir)
    rng: random.Generator = random.default_rng()

    # collect experience 'till the agent can learn
    while not agent.can_learn():
        idx, (key, value) = conn.recv()
        if key == 'buffer':
            agent.load(idx, value)
        elif key == 'except':
            print(value[1])
            raise value[0]
    conn.send(('send_fitness', None))
    # start the learning loop
    for t in count(1):
        agent.learn(t, writer, rng, LEARNER_DEVICE)
        # update the actor
        if not t % ACTOR_UPDATE_INTERVAL:
            conn.send(('net', state_dict_to_cpu(agent.net_state_dict())))
        if not t % SAVE_INTERVAL:
            torch.save(agent.net_state_dict(), 'models/' + MODEL_FILE)
        # collect experience
        if conn.poll(0):
            idx, (key, value) = conn.recv()
            if key == 'buffer':
                agent.load(idx, value)
            elif key == 'fitness':
                writer.add_scalar('Fitness', value, t)
            elif key == 'except':
                print(value[1])
                raise value[0]


def main():
    actor_conns, learner_conns = distrib.MultiPipe(NUM_ACTORS)
    actor_epsilons = BASE_EPSILON**(1 + np.arange(NUM_ACTORS) * EPSILON_ALPHA /
                                    (NUM_ACTORS - 1))
    actor_handles = [
        mp.Process(target=actor_proc, args=(conn, epsilon), daemon=True)
        for conn, epsilon in zip(actor_conns, actor_epsilons)
    ]
    for actor_handle in actor_handles:
        actor_handle.start()
    learner_proc(distrib.MultiConnection(learner_conns))


if __name__ == '__main__':
    main()
