import numpy as np

def init_logger(env):
    logger = Logger(env)
    return logger

class Logger():
    def __init__(self, env): 
        super(Logger, self).__init__()
        self.keys = env.agents
        self.epi_rewards = self._clear_dict_as_list(self.keys)
        self.rewards = self._clear_dict(self.keys)
        self.losses = self._clear_dict_as_list(self.keys)
        self.avg_window = 5  # average over the past

        self._create_log_dir()

    def _create_log_dir(self):
        pass

    def _clear_dict(self, keys, v=0.):
        return {a:v for a in keys}

    def _clear_dict_as_list(self, keys):
        return {a:[] for a in keys}

    def log_reward(self, reward):
        for k, r in zip(self.rewards.keys(), reward):
            self.rewards[k] += r

    def log_episode_reward(self,):
        for k, v in self.rewards.items():
            self.epi_rewards[k].append(v)
        self.rewards =  self._clear_dict(self.keys)

    def log_loss(self, loss):
        for k, l in zip(self.losses.keys(), loss):
            self.losses[k].append(l)

    def print(self, epi):
        print(f'Episode: {epi}')
        for k in self.keys:
            print(f"{k}: \
                episode reward: {np.mean(self.epi_rewards[k][-self.avg_window:]):.4f}, \
                loss: {np.mean(self.losses[k][-self.avg_window:]):.4f}")
