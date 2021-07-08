import numpy as np
import torch
from utils.logger import init_logger

def run_agent_single_episode(env, args, model, agent_ids):
    for i in agent_ids:
        model.agents[i].eval()
    observation = env.reset()
    r=0        
    for s in range(args.max_steps_per_episode):
        action = model.choose_action(agent_ids, np.squeeze(observation))  # squeeze list of obs for multiple agents (only one here)
        new_observation, reward, done, info = env.step(action) # unsqueeze
        r=r+reward[0]  # squeeze
        
        observation = new_observation

        if np.any(done):
            break

    return r

def run_agents_n_episodes(env, args, model):
    avg_score = []
    for epi in range(args.algorithm_spec['rollout_episodes_per_selection']):
        agent_rewards = []
        for agent_id in range(model.num_agents):
            agent_rewards.append(run_agent_single_episode(env, args, model, [agent_id]))
        avg_score.append(agent_rewards)
    avg_score = np.mean(np.vstack(avg_score).T, axis=-1)

    return avg_score

def rollout_ga(env, model, args):
    #disable gradients as we will not use them
    torch.set_grad_enabled(False)
    logger = init_logger(env, args)

    if args.marl_method == 'selfplay':
        """ Self-play with genetic algorithm, modified from https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ga_selfplay.py
            Difference: use the torch model like standard RL policies rather than a handcrafted model.
        """
        winning_streak = [0] * model.number_agents # store the number of wins for this agent (including mutated ones)

        for generation in range(args.algorithm_spec['max_generations']): # the 'generation' here is rollout of one agent
            selected_agent_ids = np.random.choice(model.number_agents, len(env.agents), replace=False)
            first_agent_reward = run_agent_single_episode(env, args, model, selected_agent_ids)
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
                model.save_model(logger.model_dir+str(generation), best_agent_id=record_holder)
    
    else:
        """ Single-agent self-play, modified from https://github.com/paraschopra/deepneuroevolution/blob/master/openai-gym-cartpole-neuroevolution.ipynb
            Difference: the add_elite step is removed, so every child is derived with mutation.
        """
        for generation in range(args.algorithm_spec['max_generations']): # the 'generation' here is rollout of all agents
            # return rewards of agents
            rewards = run_agents_n_episodes(env, args, model) #return average of 3 runs

            # sort by rewards
            sorted_parent_indexes = np.argsort(rewards)[::-1][:args.algorithm_spec['top_limit']] #reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order

            top_rewards = []
            for best_parent in sorted_parent_indexes:
                top_rewards.append(rewards[best_parent])

            print("Generation ", generation, " | Mean rewards: ", np.mean(rewards), " | Mean of top 5: ",np.mean(top_rewards[:5]))
            print("Top ",args.algorithm_spec['top_limit']," scores", sorted_parent_indexes)
            print("Rewards for top: ",top_rewards)
            
            # setup an empty list for containing children agents
            children_agents = model.return_children(sorted_parent_indexes)

            # kill all agents, and replace them with their children
            model.agents = children_agents

            if generation % args.log_interval == 0 and generation > 0: 
                model.save_model(logger.model_dir+str(generation), best_agent_id=sorted_parent_indexes[0])
