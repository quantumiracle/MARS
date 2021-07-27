import numpy as np
from datetime import datetime
import os
import json
from torch.utils.tensorboard import SummaryWriter
from utils.typing import Union, Dict, Any, List, ConfigurationDict


class TestLogger():
    """ The logger used for test mode.

    :param env: environment object
    :type env: object
    :param args: arguments
    :type args: dict
    """
    def __init__(self, env, args: ConfigurationDict) -> None:
        super(TestLogger, self).__init__()
        # if using parallel environment, env.agents is list of list,
        # we flatten it in to a simple list. For example, it changes
        # [['(env1)player1', '(env1)player2'], ['(env2)player1', '(env2)player2']]
        # to be ['env1_player1', 'env1_player2', 'env2_player1', 'env2_player2'].
        if all(isinstance(i, list) for i in env.agents):
            self.keys = [
                f'env{env_id}_' + item
                for env_id, sublist in enumerate(env.agents)
                for item in sublist
            ]
        else:
            self.keys = env.agents

        self.avg_window = args.log_avg_window  # average over the past
        self.epi_rewards = self._clear_dict_as_list(self.keys)
        self.rewards = self._clear_dict(self.keys)
        self.epi_length = []
        self.current_episode = 0
        self.model_dir = None
        self.additional_logs = []

    def _create_dirs(self, *args):
        pass

    def _clear_dict(self, keys: List[str], v: float = 0.) -> Dict[str, Any]:
        return {a: v for a in keys}

    def _clear_dict_as_list(self, keys: List[str]) -> Dict[str, Any]:
        return {a: [] for a in keys}

    def log_reward(
        self,
        reward: List[float],
    ) -> None:
        for k, r in zip(self.rewards.keys(), reward):
            self.rewards[k] += r

    def log_episode_reward(
        self,
        step: int,
    ) -> None:
        for k, v in self.rewards.items():
            self.epi_rewards[k].append(v)
        self.rewards = self._clear_dict(self.keys)
        self.epi_length.append(step)
        self.current_episode += 1

    def log_loss(self, *args):
        pass

    def print_and_save(self, *args):
        """ Print out information only since it usually does not require
        to save the logging in test mode. """
        print(
            f'Episode: {self.current_episode}, avg. length {np.mean(self.epi_length[-self.avg_window:])}'
        )
        for k in self.keys:
            print(f"{k}: \
                episode reward: {np.mean(self.epi_rewards[k][-self.avg_window:]):.4f}"
                  )

        if len(self.additional_logs) > 0:
            for log in self.additional_logs:
                print(log)
            self.additional_logs = []


class Logger(TestLogger):
    """ The standard logger used for multi-agent training.

    :param env: environment object
    :type env: object
    :param args: arguments
    :type args: dict
    """
    def __init__(self, env, args: ConfigurationDict) -> None:
        super().__init__(env, args)
        self.losses = self._clear_dict_as_list(self.keys)

        self._create_dirs(args)
        self.writer = SummaryWriter(self.runs_dir)
        # save params data
        json.dump(args, open(self.log_dir + "params.json", 'w'))

    def _create_dirs(self, args: ConfigurationDict) -> None:
        """ Create saving directories for:
        (1) logging;
        (2) tensorboard running information;
        (3) models.

        :param args: arguments
        :type args: dict
        """
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d%H%M%S")
        post_fix = f"{args.env_type}_{args.env_name}_{args.marl_method}_{args.algorithm}_{dt_string}/"

        self.log_dir = f'../data/log/' + post_fix
        self.runs_dir = f'../data/tensorboard/' + post_fix
        self.model_dir = f'../model/' + post_fix
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.runs_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def log_episode_reward(self, step: int) -> None:
        for k, v in self.rewards.items():
            self.epi_rewards[k].append(v)
            self.writer.add_scalar(f"Episode Reward/{k}",
                                   self.epi_rewards[k][-1],
                                   self.current_episode)
        self.rewards = self._clear_dict(self.keys)
        self.epi_length.append(step)
        self.current_episode += 1

    def log_loss(self, loss: List[float]) -> None:
        for k, l in zip(self.losses.keys(), loss):
            self.losses[k].append(l)
            self.writer.add_scalar(f"RL Loss/{k}", self.losses[k][-1],
                                   self.current_episode)

    def print_and_save(self):
        """ Print out information and save the logging data. """
        print(
            f'Episode: {self.current_episode}, avg. length {np.mean(self.epi_length[-self.avg_window:])}'
        )
        for k in self.keys:
            print(f"{k}: \
                episode reward: {np.mean(self.epi_rewards[k][-self.avg_window:]):.4f}, \
                loss: {np.mean(self.losses[k][-self.avg_window:]):.4f}")

        if len(self.additional_logs) > 0:
            for log in self.additional_logs:
                print(log)
            self.additional_logs = []

        # save process data
        process_data = {
            'episode_reward': self.epi_rewards,
            'loss': self.losses,
            'episode_length': self.epi_length,
        }
        json.dump(process_data, open(self.log_dir + "process.json", 'w'))

        # read the data with:
        # data = json.load( open(self.log_dir+"process.json"))


class DummyLogger(Logger):
    """ The logger used for single agent.

    :param env: environment object
    :type env: object
    :param args: arguments
    :type args: dict
    """
    def __init__(self, env, args: ConfigurationDict) -> None:
        super().__init__(env, args)
        self.avg_window = args.log_avg_window
        self.reward = 0
        self.current_episode = 0
        self.epi_rewards = []
        self.epi_length = []

    def log_reward(self, reward: float):
        self.reward += reward

    def log_episode_reward(self,
                           step: int,
                           episode_reward: Union[float, None] = None):
        if episode_reward is not None:
            self.epi_rewards.append(episode_reward)
        else:
            self.epi_rewards.append(self.reward)
        self.epi_length.append(step)
        self.reward = 0
        self.current_episode += 1

    def print_and_save(self):
        # print out info
        print(
            f'Episode: {self.current_episode}, avg. reward: {np.mean(self.epi_rewards[-self.avg_window:]):.4f}, avg. length {np.mean(self.epi_length[-self.avg_window:])}'
        )

        if len(self.additional_logs) > 0:
            for log in self.additional_logs:
                print(log)
            self.additional_logs = []

        # save process data
        process_data = {
            'episode_reward': self.epi_rewards,
            'episode_length': self.epi_length,
        }
        json.dump(process_data, open(self.log_dir + "process.json", 'w'))


def init_logger(
        env,
        args: ConfigurationDict) -> Union[DummyLogger, TestLogger, Logger]:
    """A function to initiate a proper logger.

    :param env: environment object
    :type env: object
    :param args: arguments
    :type args: dict
    :return: logger
    :rtype: object
    """
    if args.algorithm == 'GA':
        logger = DummyLogger(env, args)
    elif args.test:
        logger = TestLogger(env, args)
    else:
        logger = Logger(env, args)
    return logger