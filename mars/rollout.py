import numpy as np
from numpy.lib.arraysetops import isin
import torch
import time
from .utils.logger import init_logger
from .utils.typing import Tuple, List, ConfigurationDict
from .marl import init_meta_learner
from mars.utils.common import SelfplayBasedMethods, MetaStrategyMethods, MetaStepMethods


def rollout(env, model, args: ConfigurationDict, save_id='0') -> None:
    """
    Function to rollout the interaction of agents and environments.

    Due to the strong heterogeneity of Genetic algorithm and Reinforcement Learning
    algorithm, the function is separeted into two types. 
    """
    if args.algorithm == 'GA':
        rollout_ga(env, model, save_id, args)
    else:
        rollout_normal(env, model, save_id, args)


def eval(env, model, eval_logger, at_epi, args):
    obs = env.reset()
    greedy_list = model.number_of_agents * [True]  # greedy evaluation for all agents
    episode_reward = {a: 0 for a in env.agents}
    
    ## Rollout
    for step in range(args.max_steps_per_episode):
        obs_to_store = obs.swapaxes(0, 1) if args.num_envs > 1 else obs  # transform from (envs, agents, dim) to (agents, envs, dim)
        action_ = model.choose_action(
            obs_to_store, greedy_list)  # action: (agent, env, action_dim)

        ## Action processing
        if isinstance(action_, tuple): # Nash PPO
            (a, info) = action_  # shape: (agents, envs, dim)
            action_to_store = a
            other_info = info
        elif any(isinstance(a_, tuple) for a_ in action_):  # exploitation with PPO
            action_to_store, other_info = [], []
            for a_ in action_:  # loop over agent
                if isinstance(a_, tuple): # action item contains additional information
                    (a, info) = a_
                    action_to_store.append(a)
                    other_info.append(info)
                else:
                    action_to_store.append(a_)
                    other_info.append(None)
        else:
            action_to_store = action_
            other_info = None
        if args.num_envs > 1:
            action = np.array(action_to_store).swapaxes(0, 1)  # transform from (agents, envs, dim) to (envs, agents, dim)
        else:
            action = action_to_store

        ## Environment step
        obs_, reward, done, info = env.step(action)  # required action shape: (envs, agents, dim)
        obs = obs_
        for k, v in zip(env.agents, reward):
            episode_reward[k] += v
        if np.any(done):  # if any player in a game is done, the game episode done; may not be correct for some envs
            break

    ## Logging
    eval_logger.extr_logs['episode'].append(at_epi)
    for k, v in zip(env.agents, reward):
        eval_logger.extr_logs['episode_reward'][k].append(episode_reward[k])

def rollout_normal(env, model, save_id, args: ConfigurationDict) -> None:
    """Function to rollout experience as interaction of agents and environments, in
    a typical manner of reinforcement learning. 

    :param env: environment instance
    :type env: object
    :param model: the multi-agent model containing models for all agents
    :type model: MultiAgent
    :param args: arguments
    :type args: ConfigurationDict
    """
    ## Initialization
    print("Arguments: ", args)
    overall_steps = 0
    logger = init_logger(env, save_id, args)
    if args.exploit:
        logger.add_extr_log('eval')
        logger.extr_logs = {}
        logger.extr_logs['episode'] = []
        logger.extr_logs['episode_reward'] = {a: [] for a in env.agents}
    meta_learner = init_meta_learner(logger, args) if not args.test and not args.exploit else None

    ## Rollout
    for epi in range(args.max_episodes):
        obs = env.reset()
        for step in range(args.max_steps_per_episode):
            overall_steps += 1
            obs_to_store = obs.swapaxes(0, 1) if args.num_envs > 1 else obs  # transform from (envs, agents, dim) to (agents, envs, dim)
            with torch.no_grad():
                action_ = model.choose_action(
                    obs_to_store)  # action: (agent, env, action_dim)
            if overall_steps % 100 == 0: # do not need to do this for every step
                model.scheduler_step(overall_steps)
            
            ## Action processing
            if isinstance(action_, tuple): # Nash PPO: action item contains additional information like log probability
                (a, info) = action_  # shape: (agents, envs, dim)
                action_to_store = a
                other_info = info
            elif any(isinstance(a_, tuple) for a_ in action_):  # exploitation with PPO
                action_to_store, other_info = [], []
                for a_ in action_:  # loop over agent
                    if isinstance(a_, tuple): # action item contains additional information
                        (a, info) = a_
                        action_to_store.append(a)
                        other_info.append(info)
                    else:
                        action_to_store.append(a_)
                        other_info.append(None)
            else:
                action_to_store = action_
                other_info = None
            if args.num_envs > 1:
                action = np.array(action_to_store).swapaxes(0, 1)  # transform from (agents, envs, dim) to (envs, agents, dim)
            else:
                action = action_to_store

            obs_, reward, done, info = env.step(action)  # required action shape: (envs, agents, dim)

            if args.render:
                env.render()

            ## Storage information processing
            if args.num_envs > 1:  # transform from (envs, agents, dim) to (agents, envs, dim)
                obs__to_store = obs_.swapaxes(0, 1)
                reward_to_store = reward.swapaxes(0, 1)
                done_to_store = done.swapaxes(0, 1)
            else:
                obs__to_store = obs_
                reward_to_store = reward
                done_to_store = done

            if other_info is None:
                sample = [  # each item has shape: (agents, envs, dim)
                    obs_to_store, action_to_store, reward_to_store,
                    obs__to_store, done_to_store
                ]
            else:
                other_info_to_store = other_info # no need to swap axis, shape already: (agents, envs, dim)
                sample = [
                    obs_to_store, action_to_store, reward_to_store,
                    obs__to_store, other_info_to_store, done_to_store
                ]
            if other_info is not None or model.nan_filter(sample):  # store sample only if it is valid; cannot filter out None in other info
                model.store(sample)

            obs = obs_
            logger.log_reward(np.array(reward).reshape(-1))
            loss = None

            # Non-epsodic update of the model
            if not args.algorithm_spec['episodic_update'] and \
                 model.ready_to_update and overall_steps > args.train_start_frame:
                if args.update_itr >= 1:
                    avg_loss = []
                    for _ in range(args.update_itr):
                        loss, infos = model.update(
                        )
                        avg_loss.append(loss)
                    loss = np.mean(avg_loss, axis=0)
                elif overall_steps * args.update_itr % 1 == 0:
                    loss, infos = model.update()
                if loss is not None:
                    logger.log_loss(loss)
                    logger.log_info(infos)

            ## done break: needs to go after everything else， including the update
            if np.any(
                    done
            ):  # if any player in a game is done, the game episode done; may not be correct for some envs
                break

        ## Epsodic update of the model
        if model.ready_to_update:
            if args.algorithm_spec['episodic_update']:
                loss, infos = model.update()
                logger.log_loss(loss)
                logger.log_info(infos)
            
            if meta_learner is not None and args.marl_method in MetaStepMethods:
                meta_learner.step(
                    model, logger, env, args
                )  # metalearner for selfplay need just one step per episode
        
        ## Meta-learner step (episodic)
        if args.marl_method in MetaStrategyMethods and (args.test or args.exploit):
            # only methods in MetaStrategyMethods (subset of MetaStepMethods) during exploitation
            # requires step()
            model.meta_learner.step()  # meta_learner as the agent to be tested/exploited
        logger.log_episode_reward(step)

        ## Evaluation during exploiter training
        if epi % args.log_interval == 0:
            if args.exploit:
                eval(env, model, logger, epi, args)

            logger.print_and_save()

        ## Model saving and logging
        if epi % args.save_interval == 0 \
            and logger.model_dir is not None:
            model.save_model(logger.model_dir+f'{epi}')

        # if epi % args.save_interval == 0 \
        #     and not args.marl_method in MetaStepMethods \
        #     and logger.model_dir is not None:
        #     model.save_model(logger.model_dir+f'{epi}')

        if meta_learner is not None \
            and epi % args.save_interval == 0 \
            and args.marl_method in MetaStrategyMethods:
            meta_learner.save_model()
            # meta_learner.save_model(logger.model_dir+f'{epi}_')
        
### Genetic algorithm uses a different way of rollout. ###

def run_agent_single_episode(env, args: ConfigurationDict, model,
                             agent_ids: List[int]) -> Tuple[float, int]:
    for i in agent_ids:
        model.agents[i].eval()
    observation = env.reset()
    epi_r = 0
    for s in range(args.max_steps_per_episode):
        action = model.choose_action(
            agent_ids, observation
        )  # squeeze list of obs for multiple agents (only one here)
        new_observation, reward, done, info = env.step(action)  # unsqueeze
        epi_r = epi_r + reward[0]  # squeeze

        observation = new_observation

        if np.any(done):
            break
    epi_l = s
    return epi_r, epi_l


def run_agents_n_episodes(env, args: ConfigurationDict, model) -> Tuple[float, float]:
    avg_score = []
    avg_length = []
    for epi in range(args.algorithm_spec['rollout_episodes_per_selection']):
        agent_rewards = []
        for agent_id in range(model.num_agents):
            reward, length = run_agent_single_episode(env, args, model,
                                                      [agent_id])
            agent_rewards.append(reward)
            avg_length.append(length)
        avg_score.append(agent_rewards)
    avg_score = np.mean(np.vstack(avg_score).T, axis=-1)
    avg_length = np.mean(avg_length)

    return avg_score, avg_length


def rollout_ga(env, model, save_id, args: ConfigurationDict) -> None:
    """Function to rollout experience as interaction of agents and environments,
    as well as taking evolution in the agents population with genetic algorithm.
    It can work for either single-agent environment or multi-agent environments
    with a self-play scheme.

    :param env: environment instance
    :type env: object
    :param model: the multi-agent model containing the genetic algorithm model
    :type model: MultiAgent
    :param args: arguments
    :type args: ConfigurationDict
    """
    #disable gradients as we will not use them
    torch.set_grad_enabled(False)
    logger = init_logger(env, save_id, args)

    if args.marl_method == 'selfplay':
        """ Self-play with genetic algorithm, modified from https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ga_selfplay.py
            Difference: use the torch model like standard RL policies rather than a handcrafted model.
        """
        winning_streak = [
            0
        ] * model.num_agents  # store the number of wins for this agent (including mutated ones)

        for generation in range(
                args.algorithm_spec['max_generations']
        ):  # the 'generation' here is rollout of one agent
            selected_agent_ids = np.random.choice(model.num_agents,
                                                  len(env.agents),
                                                  replace=False)
            first_agent_reward, epi_length = run_agent_single_episode(
                env, args, model, selected_agent_ids)
            logger.log_episode_reward(epi_length, first_agent_reward)
            if first_agent_reward == 0:  # if the game is tied, add noise to one of the agents: the first one selected
                model.mutate(model.agents[selected_agent_ids[0]])
            elif first_agent_reward > 0:  # erase the loser, set it to the winner (first) and add some noise
                model.agents[selected_agent_ids[1]] = model.mutate(
                    model.agents[selected_agent_ids[0]])
                winning_streak[selected_agent_ids[1]] = winning_streak[
                    selected_agent_ids[0]]
                winning_streak[selected_agent_ids[0]] += 1
            else:
                model.agents[selected_agent_ids[0]] = model.mutate(
                    model.agents[selected_agent_ids[1]])
                winning_streak[selected_agent_ids[0]] = winning_streak[
                    selected_agent_ids[1]]
                winning_streak[selected_agent_ids[1]] += 1

            if generation % args.log_interval == 0 and generation > 0:
                record_holder = np.argmax(winning_streak)
                record = winning_streak[record_holder]
                logger.print_and_save()

                if generation % (10 * args.log_interval) == 0:
                    model.save_model(logger.model_dir + str(generation),
                                     best_agent_id=record_holder)

                # print(f"Generation: {generation}, record holder: {record_holder}")

    else:
        """ Single-agent self-play, modified from https://github.com/paraschopra/deepneuroevolution/blob/master/openai-gym-cartpole-neuroevolution.ipynb
            Difference: the add_elite step is removed, so every child is derived with mutation.
        """
        for generation in range(
                args.algorithm_spec['max_generations']
        ):  # the 'generation' here is rollout of all agents
            # return rewards of agents
            rewards, length = run_agents_n_episodes(
                env, args, model)  #return average of 3 runs

            # sort by rewards
            sorted_parent_indexes = np.argsort(
                rewards
            )[::-1][:args.algorithm_spec[
                'top_limit']]  #reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order

            top_rewards = []
            for best_parent in sorted_parent_indexes:
                top_rewards.append(rewards[best_parent])

            # print("Generation ", generation, " | Mean rewards: ", np.mean(rewards), " | Mean of top 5: ",np.mean(top_rewards[:5]))
            # print("Top ",args.algorithm_spec['top_limit']," scores", sorted_parent_indexes)
            # print("Rewards for top: ",top_rewards)
            logger.log_episode_reward(length, np.mean(rewards))

            # setup an empty list for containing children agents
            children_agents = model.return_children(sorted_parent_indexes)

            # kill all agents, and replace them with their children
            model.agents = children_agents

            if generation % args.log_interval == 0 and generation > 0:
                logger.print_and_save()

                if generation % (10 * args.log_interval) == 0:
                    model.save_model(logger.model_dir + str(generation),
                                     best_agent_id=sorted_parent_indexes[0])
