import argparse
import cloudpickle 
import torch
torch.multiprocessing.set_start_method('forkserver', force=True)
from multiprocessing import Process, Queue
from mars.env.import_env import make_env
from mars.rl.agents import *
from mars.rl.agents.multiagent import MultiAgent
from mars.utils.func import get_general_args, multiprocess_conf
from rolloutExperience import rolloutExperience
from updateModel import updateModel


parser = argparse.ArgumentParser(description='Arguments of the general launching script for MARS.')

### Load configurations
game_type = 'pettingzoo'
game = ['boxing_v1', 'surround_v1', 'combat_plane_v1', \
        'combat_tank_v1', 'pong_v2', 'tennis_v2', \
        'ice_hockey_v1', 'double_dunk_v2'][-3]

method = ['selfplay', 'selfplay2', 'fictitious_selfplay', \
            'fictitious_selfplay2', 'nash_dqn', 'nash_dqn_exploiter', \
            'nxdo2'][-3]   # nash_ppo are trained in train.py, cannot user here!

# method = 'nash_dqn_speed'


if __name__ == '__main__':
    args = get_general_args(game_type+'_'+game, method)
    num_envs = args.num_envs  # this will be changed to 1 later
    multiprocess_conf(args, method)

    ### Create env
    env = make_env(args)
    print(env)

    ### Specify models for each agent
    model1 = eval(args.algorithm)(env, args)
    model2 = eval(args.algorithm)(env, args)

    model = MultiAgent(env, [model1, model2], args)
    env.close()  # this env is only used for creating other intantiations

    # no need for this! tranform dictionary to bytes (serialization)
    # args = cloudpickle.dumps(args)
    # env = cloudpickle.dumps(env)  # this only works for single env, not for multiprocess vecenv
    processes = []
    print(args)

    # launch multiple sample rollout processes
    info_queue = Queue()
    for pro_id in range(num_envs):  
        play_process = Process(target=rolloutExperience, args = (model, info_queue, args, pro_id))
        play_process.daemon = True  # sub processes killed when main process finish
        processes.append(play_process)

    # # launch update process (single or multiple)
    default_id = '0'
    update_process = Process(target=updateModel, args= (model, info_queue, args, default_id))
    update_process.daemon = True
    processes.append(update_process)

    [p.start() for p in processes]
    while all([p.is_alive()for p in processes]):
        pass
