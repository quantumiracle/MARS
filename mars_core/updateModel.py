import numpy as np
from numpy.lib.arraysetops import isin
import torch
import time
from utils.logger import init_logger
from utils.typing import Tuple, List, ConfigurationDict
from marl.meta_learner import init_meta_learner


def updateModel(env, model, args: ConfigurationDict, save_id='0') -> None:
    """
    Function to rollout the interaction of agents and environments.

    Due to the strong heterogeneity of Genetic algorithm and Reinforcement Learning
    algorithm, the function is separeted into two types. 
    """
    if args.algorithm == 'GA':
        update_ga(env, model, save_id, args)  ## TODO
    else:
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
    overall_steps = 0
    logger = init_logger(env, save_id, args)
    meta_learner = init_meta_learner(logger, args)
    for itr in range(1000000):
        if model.ready_to_update():
            loss = model.update()

        if (itr+1) % 1000 == 0:
            meta_learner.step(
                model, logger, env, args
            )  # metalearner for selfplay need just one step per episode
        logger.log_episode_reward(itr)

        if itr % 1000*args.log_interval == 0:
            logger.print_and_save()
        if itr % 1000*args.save_interval == 0 \
        and not args.marl_method in ['selfplay', 'selfplay2', 'fictitious_selfplay', 'fictitious_selfplay2', 'nxdo', 'nxdo2'] \
        and logger.model_dir is not None:
            model.save_model(logger.model_dir+f'{itr}')


def update_ga(env, model, save_id, args: ConfigurationDict) -> None:
    pass