"""
This script is for testing the models trained in 
multi-agent setting with single-agent env (Gym),
not for testing single-agent models.
"""

from mars.env.import_env import make_env
from mars.rollout import rollout
from mars.rl.agents import *
from mars.rl.agents.multiagent import MultiAgent
from mars.utils.func import get_model_path
from mars.utils.args_parser import get_args

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

def launch():
    args = get_args()
    print(args)
    env = args.env
    method = args.marl_method

    ## Change/specify some arguments if necessary
    args.max_episodes = 1000
    args.multiprocess = False
    args.against_baseline = False
    args.test = True
    args.exploit = False
    # args.render = True
    folder = f'./data/model/{args.load_id}/{env}_{method}/'

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

    if args.save_id is not None:
        rollout(env, model, args, save_id = args.load_id+'_test') # last arg is save path


if __name__ == '__main__':
    launch()