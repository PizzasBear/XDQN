import torch.multiprocessing as mp
from multiprocessing.connection import Connection
from multiprocessing.synchronize import RLock
from threading import Thread
from typing import List, Tuple, Optional, Union
import numpy as np


def router(receiver: Connection, sender: Connection, lock: RLock, idx: int):
    i = 0
    while True:
        message = receiver.recv()
        with lock:
            sender.send((idx, message))
            i = i + 1


class MultiConnection:
    __slots__ = 'connections', 'receiver', 'lock'

    connections: List[Tuple[Connection, Thread]]
    receiver: Connection
    lock: RLock

    def __init__(self, connections: List[Connection]):
        self.receiver, sender = mp.Pipe(False)
        self.lock = mp.RLock()
        self.connections = [(conn,
                             Thread(target=router,
                                    args=(conn, sender, self.lock, i),
                                    daemon=True))
                            for i, conn in enumerate(connections)]
        for _, router_thread in self.connections:
            router_thread.start()

    def send(self, obj, idx: Optional[int] = None):
        if idx is None:
            for conn, _ in self.connections:
                conn.send(obj)
        else:
            self.connections[idx][0].send(obj)

    def recv(self) -> Tuple[int, object]:
        return self.receiver.recv()

    def poll(self, timeout: Optional[float] = None) -> bool:
        return self.receiver.poll(timeout)

    def close(self):
        for conn, router_thread in self.connections:
            conn.close()
        self.receiver.close()
        self.lock.__enter__()


class MultiSender:
    __slots__ = 'connections'

    connections: List[Connection]

    def __init__(self, multi_conn: List[Connection]):
        self.connections = multi_conn

    def send(self, obj, idx: Optional[int] = None):
        if idx is None:
            for conn in self.connections:
                conn.send(obj)
        else:
            self.connections[idx].send(obj)

    def close(self):
        for conn in self.connections:
            conn.close()


class MultiReceiver:
    __slots__ = 'multi_conn'

    multi_conn: MultiConnection

    def __init__(self, multi_conn: Union[MultiConnection, List[Connection]]):
        if isinstance(multi_conn, MultiConnection):
            self.multi_conn = multi_conn
        else:
            self.multi_conn = MultiConnection(multi_conn)

    def recv(self) -> Optional[Tuple[int, object]]:
        return self.multi_conn.recv()

    def poll(self, timeout: Optional[float] = None) -> bool:
        return self.multi_conn.poll(timeout)

    def close(self):
        return self.multi_conn.close()


def MultiPipe(n: int, duplex: bool = True):
    return zip(*[mp.Pipe(duplex) for _ in range(n)])


class CloudpickleWrapper:
    __slots__ = 'obj'

    def __init__(self, obj):
        self.obj = obj

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.obj)

    def __setstate__(self, obj):
        import pickle
        self.obj = pickle.loads(self.obj)


import gym


def env_worker(make_env, conn: Connection):
    env: gym.Env = make_env()
    while True:
        key: str
        key, value = conn.recv()
        if key == 'get_spaces':
            conn.send((env.observation_space, env.action_space))
        elif key == 'step':
            obs, reward, done, _ = env.step(value)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done))
        elif key == 'reset':
            conn.send(env.reset())
        elif key == 'close':
            conn.close()
            break
        else:
            raise RuntimeError


class MPVecEnv:
    __slots__ = 'waiting', 'num', 'conns', 'workers', 'observation_space', 'action_space'
    waiting: bool
    num: int
    conns: List[Connection]
    workers: List[mp.Process]

    def __init__(self, make_env, num: int):
        self.waiting = False
        self.num = num
        self.conns, worker_conns = MultiPipe(num)
        self.workers = [
            mp.Process(target=env_worker, args=(make_env, conn), daemon=True)
            for conn in worker_conns
        ]
        for worker in self.workers:
            worker.start()

        self.conns[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.conns[0].recv()

    def reset(self):
        for conn in self.conns:
            conn.send(('reset', None))
        return np.stack([conn.recv() for conn in self.conns])

    def step_async(self, actions):
        for conn, action in zip(self.conns, actions):
            conn.send(('step', action))
        self.waiting = True

    def poll(self, timeout: Optional[float] = None) -> bool:
        return all([conn.poll(timeout) for conn in self.conns])

    def await_step(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.waiting:
            raise RuntimeError
        obs, rewards, dones = zip(*[conn.recv() for conn in self.conns])
        self.waiting = False
        obs = np.stack(obs)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool)
        return obs, rewards, dones

    def step(self, actions):
        self.step_async(actions)
        return self.await_step()

    def close(self):
        for conn in self.conns:
            if self.waiting:
                conn.recv()
            conn.send(('close', None))
        for worker in self.workers:
            worker.join()
        for conn in self.conns:
            conn.close()
