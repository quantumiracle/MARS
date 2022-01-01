import numpy as np
from numpy.lib.arraysetops import isin
import torch
import time
import cloudpickle
from mars.utils.logger import init_logger
from mars.utils.typing import Tuple, List, ConfigurationDict
from mars.marl.meta_learner import init_meta_learner


def updateModel(env, model, args: ConfigurationDict, save_id='0') -> None:
    """
    Function to rollout the interaction of agents and environments.

    Due to the strong heterogeneity of Genetic algorithm and Reinforcement Learning
    algorithm, the function is separeted into two types. 
    """
    env = cloudpickle.loads(env)
    model = cloudpickle.loads(model)
    args = cloudpickle.loads(args)
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
    max_update_itr = 1000000
    mata_update_interval = 1000
    logger = args.logger
    meta_learner = init_meta_learner(logger, args)
    for itr in range(max_update_itr):
        if model.ready_to_update():
            loss = model.update()
            logger.log_loss(loss)

        if (itr+1) % mata_update_interval == 0:
            meta_learner.step(
                model, logger, env, args
            )  # metalearner for selfplay need just one step per episode
        
        if itr % mata_update_interval*args.save_interval == 0 \
        and not args.marl_method in ['selfplay', 'selfplay2', 'fictitious_selfplay', 'fictitious_selfplay2', 'nxdo', 'nxdo2'] \
        and logger.model_dir is not None:
            model.save_model(logger.model_dir+f'{itr}')


def update_ga(env, model, save_id, args: ConfigurationDict) -> None:
    pass