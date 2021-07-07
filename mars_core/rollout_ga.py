import numpy as np
import torch
from utils.logger import init_logger

def run_agents(env, args, model):
    reward_agents = []    
    for i in range(model.num_agents):
        model.agents[i].eval()
        observation = env.reset()
        r=0        
        for s in range(args.max_steps_per_episode):
            action = model.choose_action(i, observation[0])  # squeeze list of obs for multiple agents (only one here)
            new_observation, reward, done, info = env.step([action]) # unsqueeze
            r=r+reward[0]  # squeeze
            
            observation = new_observation

            if np.any(done):
                break

        reward_agents.append(r)            
    return reward_agents

def run_agents_n_episodes(env, args, model):
    avg_score = []
    for epi in range(args.algorithm_spec['rollout_episodes_per_selection']):
        avg_score.append(run_agents(env, args, model))
    avg_score = np.mean(np.vstack(avg_score).T, axis=-1)

    return avg_score

def rollout_ga(env, args, model):
    #disable gradients as we will not use them
    torch.set_grad_enabled(False)
    logger = init_logger(env, args)

    # How many top agents to consider as parents
    top_limit = args.algorithm_spec['top_limit']

    for generation in range(args.algorithm_spec['max_generations']):
        # return rewards of agents
        rewards = run_agents_n_episodes(env, args, model) #return average of 3 runs

        # sort by rewards
        sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit] #reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order

        top_rewards = []
        for best_parent in sorted_parent_indexes:
            top_rewards.append(rewards[best_parent])

        print("Generation ", generation, " | Mean rewards: ", np.mean(rewards), " | Mean of top 5: ",np.mean(top_rewards[:5]))
        print("Top ",top_limit," scores", sorted_parent_indexes)
        print("Rewards for top: ",top_rewards)
        
        # setup an empty list for containing children agents
        children_agents = model.return_children(sorted_parent_indexes)

        # kill all agents, and replace them with their children
        model.agents = children_agents

        if generation % args.log_interval == 0 and generation > 0: 
            model.save_model(logger.model_dir+str(generation), best_agent_id=sorted_parent_indexes[0])