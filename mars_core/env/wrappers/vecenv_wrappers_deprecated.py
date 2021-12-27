import numpy as np
from multiprocessing import Process, Pipe


def worker(remote, parent_remote, env_fn_wrapper):
    """Worker class

    References:
        https://github.com/openai/baselines/tree/master/baselines/common/vec_env
    """
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            # data = {"first_0": data[0], "second_0": data[1]}  # NOTE Assuming two agents: "first_0" and "second_0"
            ob, reward, dones, info = env.step(data)
            # ob = list(ob.values())[0]  # NOTE Assuming same observations between agents
            # reward = list(reward.values())
            # dones = list(dones.values())
            # info = list(info.values())
            remote.send((ob, reward, dones, info))
        elif cmd == 'reset':
            ob = env.reset()
            # ob = list(ob.values())[0]  # NOTE Assuming same observations between agents
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            observation_spaces = list(env.observation_spaces.values())
            action_spaces = list(env.action_spaces.values())
            remote.send((observation_spaces, action_spaces))
        elif cmd == 'action_space':
            remote.send((env.action_space))
        elif cmd == 'observation_space':
            remote.send((env.observation_space))
        elif cmd == 'num_agents':
            remote.send((env.num_agents))
        elif cmd == 'agents':
            remote.send((env.agents))
        elif cmd == 'seed':
            env.seed(data)
        elif cmd == 'render':
            env.render()
        else:
            print(cmd)
            raise NotImplementedError


class VecEnv(object):
    """An abstract asynchronous, vectorized environment

    References:
        https://github.com/openai/baselines/tree/master/baselines/common/vec_env
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        # actions = np.stack(actions, axis=1)
        self.step_async(actions)
        return self.step_wait()


class CloudpickleWrapper(object):
    """Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

    References:
        https://github.com/openai/baselines/tree/master/baselines/common/vec_env
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVectorEnv(VecEnv):
    """Vectorized environment class that collects samples in parallel using subprocesses

    Args:
        env_fns (list): list of gym environments to run in subprocesses

    References:
        https://github.com/openai/baselines/tree/master/baselines/common/vec_env
    """
    def __init__(self, env_fns):
        self.env = env_fns[0]()
        self.waiting = False
        self.closed = False
        self.nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nenvs)])
        self.ps = [
            Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # If the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), self.observation_space, self.action_space)

        self.remotes[0].send(('num_agents', None))
        self.num_agents = self.remotes[0].recv()
        self.remotes[0].send(('agents', None))
        self.agents = self.remotes[0].recv()

    # def seed(self, value):
    #     for i_remote, remote in enumerate(self.remotes):
    #         remote.send(('seed', value + i_remote))

    def seed(self, value):
        for i_remote, (remote, seed) in enumerate(zip(self.remotes, value)):
            remote.send(('seed', seed))

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def render(self):
        self.remotes[0].send(('render', None))

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True

    def sample_personas(self, is_train, is_val=True, path="./"):
        return self.env.sample_personas(is_train=is_train, is_val=is_val, path=path)


class DummyVectorEnv():
    def __init__(self):
        pass
