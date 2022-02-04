import numpy as np
import random
import math

from collections import deque, namedtuple
import itertools

transition = namedtuple('transition', 'state, action, reward, next_state, is_terminal')

class ReplayBuffer_deprecated(object):
    def __init__(self, capacity, *args, **kwargs):
        self.buffer = deque(maxlen=capacity)

    def push(self, samples):
        self.buffer.extend(samples)

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def clear(self,):
        self.buffer.clear()

    def get_len(self):
        return len(self.buffer)


class ReplayBuffer(object):
    '''
    Replay Buffer class.
    warn: does not support multi-env yet if n_multi_step > 1, samples from different envs
    need to be stored in different buffers in that case.
    '''
    def __init__(self, capacity, n_multi_step, gamma):
        self.buffer = deque(maxlen=capacity)
        self.n_multi_step = n_multi_step
        self.gamma = gamma
        self.location = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, samples):
        for sample in samples:  # sample in each env
            self.buffer.append(transition(*sample))  # extend will transform transition to np.array

    def sample(self, batch_size):
        '''
        Sample batch_size memories from the buffer.
        NB: It deals the N-step DQN
        '''
        # randomly pick batch_size elements from the buffer
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []

		# for each indices
        for i in indices:
            sum_reward = 0
            states_look_ahead = self.buffer[i].next_state
            done_look_ahead = self.buffer[i].is_terminal

            # N-step look ahead loop to compute the reward and pick the new 'next_state' (of the n-th state)
            for n in range(self.n_multi_step):
                if len(self.buffer) > i+n:
                    # compute the n-th reward
                    sum_reward += (self.gamma**n) * self.buffer[i+n].reward
                    if self.buffer[i+n].is_terminal:
                        states_look_ahead = self.buffer[i+n].next_state
                        done_look_ahead = self.buffer[i+n].is_terminal
                        break
                    else:
                        states_look_ahead = self.buffer[i+n].next_state
                        done_look_ahead = self.buffer[i+n].is_terminal

            # Populate the arrays with the next_state, reward and dones just computed
            states.append(self.buffer[i].state)
            actions.append(self.buffer[i].action)
            next_states.append(states_look_ahead)
            rewards.append(sum_reward)
            dones.append(done_look_ahead)

        return np.array(states, dtype=np.float32), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states, dtype=np.float32), np.array(dones)

    def clear(self,):
        self.buffer.clear()

    def get_len(self):
        return len(self.buffer)


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