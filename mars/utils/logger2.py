import numpy as np
from datetime import datetime
import time
import os
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from .typing import Union, Dict, Any, List, ConfigurationDict


class TestLogger():
    """ The logger used for test mode.

    :param env: environment object
    :type env: object
    :param save_id: saving identification number
    :type save_id: string
    :param args: arguments
    :type args: dict
    """
    def __init__(self, env, save_id, args: ConfigurationDict) -> None:
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
        self.additional_logs = [] # additional logs are displayed but not saved
        self.extr_logs = [] # extra logs are saved but not displayed
        self.init_time = time.time()  # initialize the time logger
        self.last_time = self.init_time
        self.save_id = save_id

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

    def log_info(self, *args):
        pass

    def add_extr_log(self, extr_log_name: str):
        """ Create extra directionary for logging.
        
        """
        self.extr_log_name = extr_log_name

    def _nan_filter(self, data):
        """ Filter out np.nan in a given array.
        """
        x=np.array(data)
        x = x[~np.isnan(x)]

        return x

    def print_and_save(self, *args):
        """ Print out information only since it usually does not require
        to save the logging in test mode. """
        # print time consumption and progress
        current_time = time.time()
        time_taken = current_time - self.last_time
        self.last_time = current_time

        print(
            f'Process ID: {self.save_id}, episode: {self.current_episode}/{self.args.max_episodes} ({100*self.current_episode/self.args.max_episodes:.4f}%), \
                avg. length: {np.mean(self.epi_length[-self.avg_window:])},\
                last time consumption/overall training time: {time_taken}s / {current_time-self.init_time} s'
        )

        for k in self.keys:
            print(f"{k}: \
                episode reward: {np.mean(self.epi_rewards[k][-self.avg_window:]):.4f}"
                  )

        if len(self.additional_logs) > 0:
            for log in self.additional_logs:
                print(log)
            self.additional_logs = []

        # save extra data in another file
        if len(self.extr_logs) > 0:
            json.dump(self.extr_logs, open(self.log_dir + f"{self.extr_log_name}.json", 'w'))



class Logger(TestLogger):
    """ The standard logger used for multi-agent training.

    :param env: environment object
    :type env: object
    :param save_id: saving identification number
    :type save_id: string
    :param args: arguments
    :type args: dict
    """
    def __init__(self, env, save_id, args: ConfigurationDict) -> None:
        super().__init__(env, save_id, args)
        self.save_id = save_id
        if '-' in self.save_id: # multiprocessing, each with a logger
           [self.save_id, self.process_id] = self.save_id.split('-')
        self.losses = self._clear_dict_as_list(self.keys)

        self.post_fix = self._create_dirs(args)
        self.writer = SummaryWriter(self.runs_dir)
        # save params data
        args['add_components'] = None  # clear added components: autoproxy buffer cannot be logged
        json.dump(args, open(self.log_dir + "params.json", 'w'))

        self.args = args
        self.current_itr = 0

    def _create_dirs(self, args: ConfigurationDict) -> None:
        """ Create saving directories for:
        (1) logging;
        (2) tensorboard running information;
        (3) models.

        :param args: arguments
        :type args: dict
        """
        post_fix = f"{args.env_type}_{args.env_name}_{args.marl_method}/"
        self.log_dir = f'./{args.save_path}/data/log/{self.save_id}/' + post_fix
        self.runs_dir = f'./{args.save_path}/data/tensorboard/{self.save_id}/' + post_fix
        self.model_dir = f'./{args.save_path}/data/model/{self.save_id}/' + post_fix
        print(f'Save models to : {os.path.abspath(self.model_dir)}. \n Save logs to: {os.path.abspath(self.log_dir)}.')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.runs_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        return post_fix

    def log_episode_reward(self, step: int) -> None:
        for k, v in self.rewards.items():
            self.epi_rewards[k].append(v)
            self.writer.add_scalar(f"Process ID: {self.process_id}, episode Reward/{k}",
                                   self.epi_rewards[k][-1],
                                   self.current_episode)

        self.rewards = self._clear_dict(self.keys)
        self.epi_length.append(step)
        self.current_episode += 1

    def log_loss(self, loss: List[float]) -> None:
        for k, l in zip(self.losses.keys(), loss):
            self.losses[k].append(l)
            self.writer.add_scalar(f"Process ID: {self.process_id}, RL Loss/{k}", 
                                   self.losses[k][-1],
                                   self.current_itr)
        self.current_itr += 1  # update iteration

    def log_info(self, infos: List[dict]) -> None:
        for i, info in enumerate(infos):
            if len(info)>0: # valid learner
                for k, v in info.items():
                    self.writer.add_scalar(f"Process ID: {self.process_id}, Metric_{i}/{k}", torch.mean(v), np.sum(self.epi_length))


    def print_and_save(self):
        """ Print out information and save the logging data. """
        # print time consumption and progress
        current_time = time.time()
        time_taken = current_time - self.last_time
        self.last_time = current_time
        if  len(self.losses[self.keys[0]])>0:  # non-empty
            print(
                f'Update itr: {self.current_itr}/{self.args.max_update_itr} ({100*self.current_itr/self.args.max_update_itr:.4f}%), \
                    last time consumption/overall running time: {time_taken:.2f}s / {current_time-self.init_time:.2f} s'
            )
            for k in self.keys:
                print(f"{k}: loss: {np.mean(self._nan_filter(self.losses[k][-self.avg_window:])):.4f}")

        elif len(self.epi_rewards[self.keys[0]])>0:  # non-empty
            print(
                f'Process ID: {self.process_id}, episode: {self.current_episode}/{self.args.max_episodes} ({100*self.current_episode/self.args.max_episodes:.4f}%), \
                    avg. length: {np.mean(self.epi_length[-self.avg_window:])},\
                    last time consumption/overall running time: {time_taken:.2f}s / {current_time-self.init_time:.2f} s'
            )
            
            for k in self.keys:
                print(f"{k}: \
                    episode reward: {np.mean(self.epi_rewards[k][-self.avg_window:]):.4f}")

        if len(self.additional_logs) > 0:
            for log in self.additional_logs:
                print(log)
            self.additional_logs = []

        # save process data
        if  len(self.losses[self.keys[0]])>0:  # non-empty
            update_data = {
                'loss': self.losses,
            }
            json.dump(update_data, open(self.log_dir + f"update_log.json", 'w'))

        if len(self.epi_rewards[self.keys[0]])>0:  # non-empty
            sample_data = {
                'episode_reward': self.epi_rewards,
                'episode_length': self.epi_length,
            }
            json.dump(sample_data, open(self.log_dir + f"sample_log.json", 'w'))

        # save extra data in another file
        if len(self.extr_logs) > 0:
            json.dump(self.extr_logs, open(self.log_dir + f"{self.extr_log_name}.json", 'w'))

class DummyLogger(Logger):
    """ The logger used for single agent.

    :param env: environment object
    :type env: object
    :param save_id: saving identification number
    :type save_id: string
    :param args: arguments
    :type args: dict
    """
    def __init__(self, env, save_id, args: ConfigurationDict) -> None:
        super().__init__(env, save_id, args)
        self.avg_window = args.log_avg_window
        self.reward = 0
        self.current_episode = 0
        self.epi_rewards = []
        self.epi_length = []
        self.save_id = save_id

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
        current_time = time.time()
        time_taken = current_time - self.last_time
        self.last_time = current_time
        print(
            f'Episode: {self.current_episode}/{self.args.max_episodes} ({100*self.current_episode/self.args.max_episodes:.4f}%), \
                avg. reward: {np.mean(self.epi_rewards[-self.avg_window:]):.4f}, \
                avg. length: {np.mean(self.epi_length[-self.avg_window:])},\
                last time consumption/overall running time: {time_taken:.2f}s / {current_time-self.init_time:.2f} s'
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
        save_id,
        args: ConfigurationDict) -> Union[DummyLogger, TestLogger, Logger]:
    """A function to initiate a proper logger.

    :param env: environment object
    :type env: object
    :param save_id: saving identification number
    :type save_id: string
    :param args: arguments
    :type args: dict
    :return: logger
    :rtype: object
    """
    if args.algorithm == 'GA':
        logger = DummyLogger(env, save_id, args)
    elif args.test:
        logger = TestLogger(env, save_id, args)
    else:
        logger = Logger(env, save_id, args)
    return logger