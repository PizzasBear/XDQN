import multiprocessing as mp
from typing import List, Tuple, Optional, Union


class MultiConnection:
    __slots__ = 'connections'

    connections: List[mp.connection.Connection]

    def __init__(self, connections: List[mp.connection.Connection]):
        self.connections = connections

    def send(self, obj, idx: Optional[int] = None):
        if idx is None:
            for conn in self.connections:
                conn.send(obj)
        else:
            self.connections[idx].send(obj)

    def recv(self) -> Optional[Tuple[int, object]]:
        for i, conn in enumerate(self.connections):
            if conn.poll(0):
                return i, conn.recv()

    def poll(self) -> List[int]:
        return [i for i, conn in enumerate(self.connections) if conn.poll(0)]

    def poll_any(self) -> bool:
        for conn in self.connections:
            if conn.poll(0):
                return True
        return False

    def close(self):
        for conn in self.connections:
            conn.close()


class MultiSender:
    __slots__ = 'multi_conn'

    multi_conn: MultiConnection

    def __init__(self, multi_conn: Union[MultiConnection,
                                         List[mp.connection.Connection]]):
        if isinstance(multi_conn, MultiConnection):
            self.multi_conn = multi_conn
        else:
            self.multi_conn = MultiConnection(multi_conn)

    def send(self, obj, idx: Optional[int] = None):
        self.multi_conn.send(obj, idx)

    def close(self):
        return self.multi_conn.close()


class MultiReceiver:
    __slots__ = 'multi_conn'

    multi_conn: MultiConnection

    def __init__(self, multi_conn: Union[MultiConnection,
                                         List[mp.connection.Connection]]):
        if isinstance(multi_conn, MultiConnection):
            self.multi_conn = multi_conn
        else:
            self.multi_conn = MultiConnection(multi_conn)

    def recv(self) -> Optional[Tuple[int, object]]:
        return self.multi_conn.recv()

    def poll(self) -> List[int]:
        return self.multi_conn.poll()

    def poll_any(self) -> bool:
        return self.multi_conn.poll_any()

    def close(self):
        return self.multi_conn.close()


def MultiPipe(
    n: int,
    duplex: bool = True
) -> Tuple[List[mp.connection.Connection], List[mp.connection.Connection]]:
    receivers = []
    senders = []
    for _ in range(n):
        receiver, sender = mp.Pipe(duplex)
        receivers.append(receiver)
        senders.append(sender)
    return receivers, senders
