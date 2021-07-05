import numpy as np
from datetime import datetime
import os
import json
from torch.utils.tensorboard import SummaryWriter


def init_logger(env, args):
    logger = Logger(env, args)
    return logger


class Logger():
    def __init__(self, env, args):
        super(Logger, self).__init__()
        self.keys = env.agents
        self.epi_rewards = self._clear_dict_as_list(self.keys)
        self.rewards = self._clear_dict(self.keys)
        self.losses = self._clear_dict_as_list(self.keys)
        self.avg_window = args.log_avg_window  # average over the past
        self.epi_length = []
        self.current_episode = 0

        self._create_dirs(args)
        self.writer = SummaryWriter(self.runs_dir)
        # save params data
        json.dump(args, open(self.log_dir + "params.json", 'w'))

    def _create_dirs(self, args):
        """
        Create saving directories for:
        * logging
        * tensorboard running information
        * models
        """
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y%H%M%S")
        post_fix = f"{args.env_type}_{args.env_name}_marl_method_{dt_string}/"

        self.log_dir = f'../data/log/' + post_fix
        self.runs_dir = f'../data/tensorboard/' + post_fix
        self.model_dir = f'../model/' + post_fix
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.runs_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def _clear_dict(self, keys, v=0.):
        return {a: v for a in keys}

    def _clear_dict_as_list(self, keys):
        return {a: [] for a in keys}

    def log_reward(self, reward):
        for k, r in zip(self.rewards.keys(), reward):
            self.rewards[k] += r

    def log_episode_reward(self, step):
        for k, v in self.rewards.items():
            self.epi_rewards[k].append(v)
            self.writer.add_scalar(f"Episode Reward/{k}",
                                    self.epi_rewards[k][-1],
                                    self.current_episode)
        self.rewards = self._clear_dict(self.keys)
        self.epi_length.append(step)
        self.current_episode += 1


    def log_loss(self, loss):
        for k, l in zip(self.losses.keys(), loss):
            self.losses[k].append(l)
            self.writer.add_scalar(f"RL Loss/{k}", self.losses[k][-1],
                                    self.current_episode)

    def print_and_save(self):
        # print out info
        print(
            f'Episode: {self.current_episode}, avg. length {np.mean(self.epi_length[-self.avg_window:])}'
        )
        for k in self.keys:
            print(f"{k}: \
                episode reward: {np.mean(self.epi_rewards[k][-self.avg_window:]):.4f}, \
                loss: {np.mean(self.losses[k][-self.avg_window:]):.4f}")

        # save process data
        process_data = {
            'episode reward': self.epi_rewards,
            'loss': self.losses,
        }
        json.dump(process_data, open(self.log_dir + "process.json", 'w'))

        # read the data with:
        # data = json.load( open(self.log_dir+"process.json"))

