import numpy as np
import random
import math

from collections import deque, namedtuple
import itertools

transition = namedtuple('transition', 'state, action, reward, next_state, is_terminal')

def ReplayBuffer(capacity, n_multi_step, gamma, num_envs, batch_size):
    """ Function to choose a proper replay buffer"""
    if n_multi_step == 1:
        return SimpleReplayBuffer(capacity)  # this one is simple and quick
    else:
        return MultiStepReplayBuffer(capacity, n_multi_step, gamma, num_envs, batch_size)

class SimpleReplayBuffer(object):
    """Replay Buffer class for one-step return. This is the
    simplest and quicker setting.

    :param capacity: number of samples stored in total
    :type capacity: int
    :return: None
    :rtype: None
    """
    def __init__(self, capacity, *args, **kwargs):
        self.buffer = deque(maxlen=capacity)

    def push(self, samples):
        self.buffer.extend(samples)

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state, dtype=np.float32), action, np.array(reward, dtype=np.float32), np.array(next_state, dtype=np.float32), done

    def clear(self,):
        self.buffer.clear()

    def get_len(self):
        return len(self.buffer)

class MultiStepReplayBuffer(object):
    """Replay Buffer class for multi-step return.

    :param capacity: number of samples stored in total
    :type capacity: int
    :param n_multi_step: n for n-step return
    :type n_multi_step: int
    :param gamma: reward discount factor for calculating n-step return
    :type gamma: float
    :param num_envs: number of environments; note that when num_envs > 1, samples need to be stored separately per environment to calculate n-step return without mutual interference
    :type num_envs: int
    :param batch_size: batch size for update
    :type batch_size: int
    :return: None
    :rtype: None
    """
    def __init__(self, capacity, n_multi_step, gamma, num_envs, batch_size):
        self.buffer = [deque(maxlen=capacity) for _ in range(num_envs)]  # list of deque
        self.n_multi_step = n_multi_step
        self.gamma = gamma
        self.num_envs = num_envs
        self.location = 0
        self.per_env_batch_sizes = [batch_size//self.num_envs for _ in range(self.num_envs)]
        if batch_size % self.num_envs != 0:  # divieded with remainder
            self.per_env_batch_sizes[0] += 1  # use one more samples in the first env buffer

    def __len__(self):
        return len(self.buffer[0])

    def push(self, samples):
        for env_idx, sample in enumerate(samples):  # sample in each env
            self.buffer[env_idx].append(transition(*sample))  # extend will transform transition to np.array

    def sample(self, batch_size):
        '''
        Sample batch_size memories from the buffer.
        NB: It deals the N-step DQN
        '''
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []
        
        for env_idx, per_env_buffer in enumerate(self.buffer):
            # randomly pick batch_size elements from the buffer
            indices = np.random.choice(len(per_env_buffer), self.per_env_batch_sizes[env_idx], replace=False)

            # for each indices
            for i in indices:
                sum_reward = 0
                states_look_ahead = per_env_buffer[i].next_state
                done_look_ahead = per_env_buffer[i].is_terminal

                # N-step look ahead loop to compute the reward and pick the new 'next_state' (of the n-th state)
                for n in range(self.n_multi_step):
                    if len(per_env_buffer) > i+n:
                        # compute the n-th reward
                        sum_reward += (self.gamma**n) * per_env_buffer[i+n].reward
                        if per_env_buffer[i+n].is_terminal:
                            states_look_ahead = per_env_buffer[i+n].next_state
                            done_look_ahead = per_env_buffer[i+n].is_terminal
                            break
                        else:
                            states_look_ahead = per_env_buffer[i+n].next_state
                            done_look_ahead = per_env_buffer[i+n].is_terminal

                # Populate the arrays with the next_state, reward and dones just computed
                states.append(per_env_buffer[i].state)
                actions.append(per_env_buffer[i].action)
                next_states.append(states_look_ahead)
                rewards.append(sum_reward)
                dones.append(done_look_ahead)

        return np.array(states, dtype=np.float32), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states, dtype=np.float32), np.array(dones)

    def clear(self,):
        for i in range(self.num_envs):
            self.buffer[i].clear()

    def get_len(self):
        return len(self.buffer[0])  # each per_env_buffer has same size


class ReservoirBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, samples):
        self.buffer.extend(samples)
    
    def sample(self, batch_size):
        # Efficient Reservoir Sampling
        # http://erikerlandson.github.io/blog/2015/11/20/very-fast-reservoir-sampling/
        n = len(self.buffer)
        reservoir = list(itertools.islice(self.buffer, 0, batch_size))
        threshold = batch_size * 4
        idx = batch_size
        while (idx < n and idx <= threshold):
            m = random.randint(0, idx)
            if m < batch_size:
                reservoir[m] = self.buffer[idx]
            idx += 1
        
        while (idx < n):
            p = float(batch_size) / idx
            u = random.random()
            g = math.floor(math.log(u) / math.log(1 - p))
            idx = idx + g
            if idx < n:
                k = random.randint(0, batch_size - 1)
                reservoir[k] = self.buffer[idx]
            idx += 1
        # state, action = zip(*random.sample(self.buffer, batch_size))
        # return np.concatenate(state), action
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def clear(self,):
        self.buffer.clear()
    
    def get_len(self):
        return len(self.buffer)

if __name__ == "__main__":
    size = 100
    buffer = ReplayBuffer(size)
    for _ in range(size):
        buffer.push([0, 0, 0, 0, 0])
    print(buffer.get_len())
    buffer.clear()
    print(buffer.get_len())
