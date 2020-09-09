import multiprocessing as mp
import dqn
import distrib


NUM_ACTORS: int = 10


def create_net():
    pass


def actor(conn: mp.connection.Connection):
    pass


def learner(sender: distrib.MultiConnection):
    pass


def main():
    actor_conns, learner_conns = distrib.MultiPipe(NUM_ACTORS)
    actor_processes = [mp.Process(target=actor, args=(conn, )) for conn in actor_conns]
    learner_proc = mp.Process(target=learner, args=(distrib.MultiConnection(learner_conns), ))
    for actor_proc in actor_processes:
        actor_proc.start()
    learner_proc.start()

    for actor_proc in actor_processes:
        actor_proc.join()
    learner_proc.join()


if __name__ == '__main__':
    main()
