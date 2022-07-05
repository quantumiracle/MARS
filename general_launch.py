### This script requires Python >= 3.7 

import argparse
import cloudpickle 
import torch
torch.multiprocessing.set_start_method('forkserver', force=True)
from multiprocessing import Process, Queue
from mars.env.import_env import make_env
from mars.rl.agents import *
from mars.rl.agents.multiagent import MultiAgent
from mars.utils.func import multiprocess_conf
from mars.rolloutExperience import rolloutExperience
from mars.updateModel import updateModel
from mars.utils.args_parser import get_args

parser = argparse.ArgumentParser(description='Arguments of the general launching script for MARS.')

def launch():
    args = get_args()
    env = args.env
    method = args.marl_method
    multiprocess_conf(args, method)

    ### Create env
    env = make_env(args)
    print(env)

    ### Specify models for each agent
    model1 = eval(args.algorithm)(env, args)
    model2 = eval(args.algorithm)(env, args)

    model = MultiAgent(env, [model1, model2], args)
    print(args)
    env.close()

    # tranform dictionary to bytes (serialization)
    # args = cloudpickle.dumps(args)
    # env = cloudpickle.dumps(env)  # this only works for single env, not for multiprocess vecenv
    processes = []

    # launch multiple sample rollout processes
    info_queue = Queue()
    for pro_id in range(1):  
        play_process = Process(target=rolloutExperience, args = (model, info_queue, args, args.save_id+'-'+str(pro_id)))
        play_process.daemon = True  # sub processes killed when main process finish
        processes.append(play_process)

    # launch update process (single or multiple)
    for pro_id in range(args.num_process):  
        update_process = Process(target=updateModel, args= (model, info_queue, args, args.save_id+'-'+str(pro_id)))
        update_process.daemon = True
        processes.append(update_process)

    [p.start() for p in processes]
    while all([p.is_alive()for p in processes]):
        pass

    [p.join() for p in processes]

        
if __name__ == '__main__':
    launch()
