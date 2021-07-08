import numpy as np
from utils.logger import init_logger
from marl.meta_learner import init_meta_learner
import torch

def rollout(env, model, args):
    if args.algorithm == 'GA':
        rollout_ga(env, model, args)
    else:
        rollout_normal(env, model, args)

def rollout_normal(env, model, args):
    print("Arguments: ", args)
    overall_steps = 0
    logger = init_logger(env, args)
    meta_learner = init_meta_learner(logger, args)
    for epi in range(args.max_episodes):
        obs = env.reset()
        for step in range(args.max_steps_per_episode):
            overall_steps += 1
            action_ = model.choose_action(obs)
            model.scheduler_step(overall_steps)
            if isinstance(action_[0], tuple): # action item contains additional information like log probability
                action, other_info = [], []
                for (a, info) in action_:
                    action.append(a)
                    other_info.append(info)
            else:
                action = action_
                other_info = None
            obs_, reward, done, info = env.step(action)

            if args.render:
                env.render()

            if other_info is None: 
                sample = [obs, action, reward, obs_, done]
            else:
                sample = [obs, action, reward, obs_, other_info, done]
            model.store(sample)

            obs = obs_
            logger.log_reward(reward)

            if np.any(
                    done
            ):  # if any player in a game is done, the game episode done; may not be correct for some envs
                logger.log_episode_reward(step)
                break

            if not args.algorithm_spec['episodic_update'] and \
                 model.ready_to_update and overall_steps > args.train_start_frame:
                if args.update_itr >= 1:
                    for _ in range(args.update_itr):
                        loss = model.update()  # only log loss for once, loss is a list
                elif overall_steps * args.update_itr % 1 == 0:
                    loss = model.update()
                if overall_steps % 1000 == 0:  # loss logging interval
                    logger.log_loss(loss)

        if model.ready_to_update:
            if args.algorithm_spec['episodic_update']:
                loss = model.update()
                logger.log_loss(loss)

            meta_learner.step(
                model,
                logger)  # metalearner for selfplay need just one step per episode

        if epi % args.log_interval == 0:
            logger.print_and_save()
            if not args.marl_method:
                model.save_model(logger.model_dir)


### Genetic algorithm uses a different way of rollout. ###

def run_agent_single_episode(env, args, model, agent_ids):
    for i in agent_ids:
        model.agents[i].eval()
    observation = env.reset()
    epi_r=0        
    for s in range(args.max_steps_per_episode):
        action = model.choose_action(agent_ids, observation)  # squeeze list of obs for multiple agents (only one here)
        new_observation, reward, done, info = env.step(action) # unsqueeze
        epi_r=epi_r+reward[0]  # squeeze
        
        observation = new_observation

        if np.any(done):
            break
    epi_l = s
    return epi_r, epi_l

def run_agents_n_episodes(env, args, model):
    avg_score = []
    avg_length = []
    for epi in range(args.algorithm_spec['rollout_episodes_per_selection']):
        agent_rewards = []
        for agent_id in range(model.num_agents):
            reward, length = run_agent_single_episode(env, args, model, [agent_id])
            agent_rewards.append(reward)
            avg_length.append(length)
        avg_score.append(agent_rewards)
    avg_score = np.mean(np.vstack(avg_score).T, axis=-1)
    avg_length = np.mean(avg_length)

    return avg_score, avg_length

def rollout_ga(env, model, args):
    #disable gradients as we will not use them
    torch.set_grad_enabled(False)
    logger = init_logger(env, args)

    if args.marl_method == 'selfplay':
        """ Self-play with genetic algorithm, modified from https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ga_selfplay.py
            Difference: use the torch model like standard RL policies rather than a handcrafted model.
        """
        winning_streak = [0] * model.num_agents # store the number of wins for this agent (including mutated ones)

        for generation in range(args.algorithm_spec['max_generations']): # the 'generation' here is rollout of one agent
            selected_agent_ids = np.random.choice(model.num_agents, len(env.agents), replace=False)
            first_agent_reward, epi_length = run_agent_single_episode(env, args, model, selected_agent_ids)
            logger.log_episode_reward(epi_length, first_agent_reward)
            if first_agent_reward == 0: # if the game is tied, add noise to one of the agents: the first one selected
                model.mutate(model.agents[selected_agent_ids[0]])
            elif first_agent_reward > 0: # erase the loser, set it to the winner (first) and add some noise
                model.agents[selected_agent_ids[1]] = model.mutate(model.agents[selected_agent_ids[0]])
                winning_streak[selected_agent_ids[1]] = winning_streak[selected_agent_ids[0]]
                winning_streak[selected_agent_ids[0]] += 1
            else:
                model.agents[selected_agent_ids[0]] = model.mutate(model.agents[selected_agent_ids[1]])
                winning_streak[selected_agent_ids[0]] = winning_streak[selected_agent_ids[1]]
                winning_streak[selected_agent_ids[1]] += 1

            if generation % args.log_interval == 0 and generation > 0: 
                record_holder = np.argmax(winning_streak)
                record = winning_streak[record_holder]
                logger.print_and_save()

                if generation % (10*args.log_interval) == 0:
                    model.save_model(logger.model_dir+str(generation), best_agent_id=record_holder)
                
                # print(f"Generation: {generation}, record holder: {record_holder}")
    
    else:
        """ Single-agent self-play, modified from https://github.com/paraschopra/deepneuroevolution/blob/master/openai-gym-cartpole-neuroevolution.ipynb
            Difference: the add_elite step is removed, so every child is derived with mutation.
        """
        for generation in range(args.algorithm_spec['max_generations']): # the 'generation' here is rollout of all agents
            # return rewards of agents
            rewards, length = run_agents_n_episodes(env, args, model) #return average of 3 runs

            # sort by rewards
            sorted_parent_indexes = np.argsort(rewards)[::-1][:args.algorithm_spec['top_limit']] #reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order

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

            if generation % (10*args.log_interval) == 0 and generation > 0: 
                logger.print_and_save()

                if generation % (10*args.log_interval) == 0:
                    model.save_model(logger.model_dir+str(generation), best_agent_id=sorted_parent_indexes[0])
