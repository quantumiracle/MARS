"""
This script is for testing the models trained in 
multi-agent setting with single-agent env (Gym),
not for testing single-agent models.
"""

from mars.env.import_env import make_env
from mars.rollout import rollout
from mars.rl.agents import *
from mars.rl.agents.multiagent import MultiAgent
from mars.utils.func import get_general_args, get_model_path, get_exploiter
import argparse
import os
parser = argparse.ArgumentParser(description='Arguments of the general launching script for MARS.')

def map_pettingzoo_to_gym(EnvNamePettingzoo):
    map_dict = {
        'boxing_v1': 'Boxing-ram-v0'
    }
    try:
        EnvNameGym = map_dict[EnvNamePettingzoo]
    except:
        print(f'No matched env in Gym for {EnvNamePettingzoo}.')
    print(f'From Pettingzoo env {EnvNamePettingzoo} to Gym env {EnvNameGym}.')
    return EnvNameGym

def launch_rollout(env, method, load_id, save_id):
    args = get_general_args(env, method)
    print(args)

    ## Change/specify some arguments if necessary
    args.max_episodes = 1000
    args.against_baseline = False
    args.test = True
    args.exploit = False
    # args.render = True
    folder = f'../data/model/{load_id}/{env}_{method}/'

    args.load_model_full_path = get_model_path(method, folder)

    # Change to test args in single-agent env
    game = env.split('_', 1)[-1]  # gametype_game_v#
    print(game)
    args.env_name = map_pettingzoo_to_gym(game)
    args.env_type = 'gym'

    ### Create env
    env = make_env(args)

    ### Specify models for each agent
    model1 = eval(args.algorithm)(env, args)

    model = MultiAgent(env, [model1], args)  

    rollout(env, model, args, save_id = load_id+'_test') # last arg is save path


if __name__ == '__main__':
    parser.add_argument('--env', type=str, default=None, help='environment')
    parser.add_argument('--method', type=str, default=None, help='method name')
    parser.add_argument('--load_id', type=str, default=None, help='identification number for loading models')
    parser.add_argument('--save_id', type=str, default=None, help='identification number for saving models')
    parser_args = parser.parse_args()
    launch_rollout(parser_args.env, parser_args.method, parser_args.load_id, parser_args.save_id)