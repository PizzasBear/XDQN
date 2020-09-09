import multiprocessing as mp
from typing import List, Tuple, Optional, Union


class MultiConnection:
    connections: List[mp.connection.Connection]

    def __init__(self, connections: List[mp.connection.Connection]):
        self.connections = connections

    def send(self, obj):
        for conn in self.connections:
            conn.send(obj)

    def recv(self) -> Optional[object]:
        for conn in self.connections:
            if conn.poll(0):
                return conn.recv()

    def poll(self) -> bool:
        return any(conn.poll(0) for conn in self.connections)


class MultiSender:
    multi_conn: MultiConnection

    def __init__(self, multi_conn: Union[MultiConnection, List[mp.connection.Connection]]):
        if isinstance(multi_conn, MultiConnection):
            self.multi_conn = multi_conn
        else:
            self.multi_conn = MultiConnection(multi_conn)

    def send(self, obj):
        self.multi_conn.send(obj)


class MultiReceiver:
    multi_conn: MultiConnection

    def __init__(self, multi_conn: Union[MultiConnection, List[mp.connection.Connection]]):
        if isinstance(multi_conn, MultiConnection):
            self.multi_conn = multi_conn
        else:
            self.multi_conn = MultiConnection(multi_conn)

    def recv(self) -> Optional[object]:
        return self.multi_conn.recv()

    def poll(self) -> bool:
        return self.multi_conn.poll()


def MultiPipe(n: int, duplex: bool = True) -> Tuple[List[mp.connection.Connection], List[mp.connection.Connection]]:
    receivers = []
    senders = []
    for _ in range(n):
        receiver, sender = mp.Pipe(duplex)
        receivers.append(receiver)
        senders.append(sender)
    return receivers, senders
