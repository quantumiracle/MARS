import numpy as np
from numpy.lib.arraysetops import isin
import torch
import time
import cloudpickle
from mars.utils.logger2 import init_logger
from mars.utils.typing import Tuple, List, ConfigurationDict
from mars.marl.meta_learner import init_meta_learner
from mars.env.import_env import make_env


def updateModel(model, args: ConfigurationDict, save_id='0') -> None:
    """
    Function to rollout the interaction of agents and environments.

    Due to the strong heterogeneity of Genetic algorithm and Reinforcement Learning
    algorithm, the function is separeted into two types. 
    """
    # tranform bytes to dictionary
    model = cloudpickle.loads(model)
    args = cloudpickle.loads(args)
    args.num_envs = 1
    env = make_env(args)
    update_normal(env, model, save_id, args)


def update_normal(env, model, save_id, args: ConfigurationDict) -> None:
    """Function to rollout experience as interaction of agents and environments, in
    a typical manner of reinforcement learning. 

    :param env: environment instance
    :type env: object
    :param model: the multi-agent model containing models for all agents
    :type model: MultiAgent
    :param args: arguments
    :type args: ConfigurationDict
    """
    print("Arguments: ", args)
    meta_update_interval = 10  # timestep interval for one meta-step
    max_update_itr = args.max_episodes * meta_update_interval
    logger = init_logger(env, save_id, args)
    meta_learner = init_meta_learner(logger, args)
    for itr in range(max_update_itr):
        if model.ready_to_update:
            loss = model.update()
            logger.log_loss(loss)

        if (itr+1) % meta_update_interval == 0:
            meta_learner.step(
                model, logger, env, args
            )  # metalearner for selfplay need just one step per episode

        if itr % (meta_update_interval*args.log_interval) == 0:
            logger.print_and_save()
        if itr % meta_update_interval*args.save_interval == 0 \
        and not args.marl_method in ['selfplay', 'selfplay2', 'fictitious_selfplay', 'fictitious_selfplay2', 'nxdo', 'nxdo2'] \
        and logger.model_dir is not None:
            model.save_model(logger.model_dir+f'{itr}')

def update_ga(env, model, save_id, args: ConfigurationDict) -> None:
    pass