import argparse
import copy
import cloudpickle 
import torch
torch.multiprocessing.set_start_method('forkserver', force=True)
from multiprocessing import Process
from multiprocessing.managers import BaseManager, NamespaceProxy

from mars.env.import_env import make_env
from mars.rl.agents import *
from mars.utils.func import get_general_args
from mars.rl.common.storage import ReplayBuffer, ReservoirBuffer
from mars.utils.common import EvaluationModelMethods
from rolloutExperience import rolloutExperience
from updateModel import updateModel

parser = argparse.ArgumentParser(description='Arguments of the general launching script for MARS.')

def multiprocess_buffer_register(ori_args, args, method):
    BaseManager.register('replay_buffer', ReplayBuffer)
    if method == 'nfsp':
        BaseManager.register('reservoir_buffer', ReservoirBuffer)
    manager = BaseManager()
    manager.start()
    args.replay_buffer = manager.replay_buffer(int(float(ori_args.algorithm_spec['replay_buffer_size'])))  
    if method == 'nfsp':
        args.reservoir_buffer = manager.reservoir_buffer(int(float(ori_args.algorithm_spec['replay_buffer_size'])))  

    return args

def launch_rollout(env, method, save_id):
    ori_args = get_general_args(env, method)
    ori_args.multiprocess = True

    ### Create env
    args = copy.copy(ori_args)
    args.num_envs = 1
    env = make_env(args)
    print(env)

    ### Specify models for each agent
    args = multiprocess_buffer_register(ori_args, args, method)
    model1 = eval(args.algorithm)(env, args)
    model2 = eval(args.algorithm)(env, args)

    if method in EvaluationModelMethods:
        args.eval_models = True
    else:
        args.eval_models = False
    model = MultiAgent(env, [model1, model2], args)
    env.close()

    # tranform dictionary to bytes (serialization)
    model = cloudpickle.dumps(model)
    args = cloudpickle.dumps(args)
    # env = cloudpickle.dumps(env)  # this only works for single env, not for multiprocess vecenv
    processes = []
    print(ori_args)

    # launch multiple sample rollout processes
    for pro_id in range(ori_args.num_envs):  
        play_process = Process(target=rolloutExperience, args = (model, args, pro_id))
        play_process.daemon = True  # sub processes killed when main process finish
        processes.append(play_process)

    # launch update process (single or multiple)
    update_process = Process(target=updateModel, args= (model, args, save_id))
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
