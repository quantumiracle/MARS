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

def launch_rollout(env, method, save_id):
    args = get_general_args(env, method)
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
    for pro_id in range(args.num_process):  
        play_process = Process(target=rolloutExperience, args = (model, info_queue, args, pro_id))
        play_process.daemon = True  # sub processes killed when main process finish
        processes.append(play_process)

    # launch update process (single or multiple)
    for pro_id in range(args.num_process):  
        update_process = Process(target=updateModel, args= (model, info_queue, args, pro_id))
        update_process.daemon = True
        processes.append(update_process)

    [p.start() for p in processes]
    while all([p.is_alive()for p in processes]):
        pass

        
if __name__ == '__main__':
    parser.add_argument('--env', type=str, default=None, help='environment')
    parser.add_argument('--method', type=str, default=None, help='method name')
    parser.add_argument('--save_id', type=str, default=None, help='identification number for each run')
    parser_args = parser.parse_args()
    launch_rollout(parser_args.env, parser_args.method, parser_args.save_id)
