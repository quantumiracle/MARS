from mars.utils.func import LoadYAML2Dict
from mars.env.import_env import make_env
from mars.rollout import rollout
from mars.rl.agents import *
from mars.rl.agents.multiagent import MultiAgent
from mars.utils.common import EvaluationModelMethods
from mars.utils.func import get_general_args
import argparse
parser = argparse.ArgumentParser(description='Arguments of the general launching script for MARS.')

def launch_rollout(env, method, save_id):
    args = get_general_args(env, method)

    ### Create env
    env = make_env(args)
    print(env)

    ### Specify models for each agent     
    model1 = eval(args.algorithm)(env, args)
    model2 = eval(args.algorithm)(env, args)

    if method in EvaluationModelMethods:
        args.eval_models = True
    else:
        args.eval_models = False
    model = MultiAgent(env, [model1, model2], args)

    ### Rollout
    rollout(env, model, args, save_id)

if __name__ == '__main__':
    parser.add_argument('--env', type=str, default=None, help='environment')
    parser.add_argument('--method', type=str, default=None, help='method name')
    parser.add_argument('--save_id', type=str, default=None, help='identification number for each run')
    parser_args = parser.parse_args()
    launch_rollout(parser_args.env, parser_args.method, parser_args.save_id)