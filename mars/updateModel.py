import cloudpickle
from mars.utils.logger2 import init_logger
from mars.utils.typing import Tuple, List, ConfigurationDict
from mars.marl import init_meta_learner
from mars.env.import_env import make_env
from mars.utils.common import SelfplayBasedMethods, MetaStrategyMethods, MetaStepMethods


def updateModel(model, info_queue, args: ConfigurationDict, save_id='0') -> None:
    """
    Function to rollout the interaction of agents and environments.

    Due to the strong heterogeneity of Genetic algorithm and Reinforcement Learning
    algorithm, the function is separeted into two types. 
    """
    # tranform bytes to dictionary
    # model = cloudpickle.loads(model) # if use this the model in different process will no longer be shared!
    # args = cloudpickle.loads(args)
    env = make_env(args)
    update_normal(env, model, info_queue, save_id, args)


def update_normal(env, model, info_queue, save_id, args: ConfigurationDict) -> None:
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
    meta_update_interval = 1000  # timestep interval for one meta-step
    max_update_itr = args.max_episodes * meta_update_interval
    args.max_update_itr = max_update_itr
    logger = init_logger(env, save_id, args)
    meta_learner = init_meta_learner(logger, args) if not args.test and not args.exploit else None
    loss = None
    for itr in range(max_update_itr):
        if model.ready_to_update:
            loss, infos = model.update()
        
        if loss is not None:
            logger.log_loss(loss)
            logger.log_info(infos)

        if meta_learner is not None \
            and args.marl_method in MetaStepMethods \
            and (itr+1) % meta_update_interval == 0:
            # meta_learner step requires episodic reward information, 
            # so receive from the experience rollout process and fill into logger
            lastest_epi_rewards = info_queue.get() # receive rollout info
            for k, r in zip(logger.epi_rewards.keys(), lastest_epi_rewards):
                logger.epi_rewards[k].append(r)
            meta_learner.step(
                model, logger, env, args
            )  # metalearner for selfplay need just one step per episode

        if args.marl_method in MetaStrategyMethods and (args.test or args.exploit):
            # only methods in MetaStrategyMethods (subset of MetaStepMethods) during exploitation
            # requires step()
            model.meta_learner.step()  # meta_learner as the agent to be tested/exploited

        if (itr+1) % (meta_update_interval*args.log_interval) == 0:
            logger.print_and_save()

        # TODO does this function really save model in training?
        # if (itr+1) % (meta_update_interval*args.save_interval) == 0 \
        #     and not args.marl_method in MetaStepMethods \
        #     and logger.model_dir is not None:
        #     # model.save_model(logger.model_dir+f'{itr+1}')
        #     model.save_model(logger.model_dir+f'{0}')  # to not save all time, but just one

        if meta_learner is not None \
            and (itr+1) % (meta_update_interval*args.save_interval) == 0 \
            and args.marl_method in MetaStrategyMethods:
            meta_learner.save_model()  # save the meta-strategy

def update_ga(env, model, save_id, args: ConfigurationDict) -> None:
    pass
