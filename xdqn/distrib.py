import torch.multiprocessing as mp
from multiprocessing.connection import Connection
from multiprocessing.synchronize import RLock
from threading import Thread
from typing import List, Tuple, Optional, Union


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

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


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

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


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

    def poll(self) -> bool:
        return self.multi_conn.poll()

    def close(self):
        return self.multi_conn.close()


def MultiPipe(
        n: int,
        duplex: bool = True) -> Tuple[List[Connection], List[Connection]]:
    receivers = []
    senders = []
    for _ in range(n):
        receiver, sender = mp.Pipe(duplex)
        receivers.append(receiver)
        senders.append(sender)
    return receivers, senders
