from utils.func import LoadYAML2Dict
from env.import_env import make_env
from rollout import rollout
from rl.algorithm import *
import argparse
parser = argparse.ArgumentParser(description='Arguments of the general launching script for MARS.')

def launch_rollout(env, method):
    prefix = 'confs/'
    postfix = '_dqn'
    yaml_file = prefix+env+'_'+method+postfix
    args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)
    
    ### Create env
    env = make_env(args)
    print(env)

    ### Specify models for each agent     
    model1 = eval(args.algorithm)(env, args)
    model2 = eval(args.algorithm)(env, args)
    # model1.fix()  # fix a model if you don't want it to learn

    if method in ['nxdo', 'nxdo2']:
        eval_env = make_env(args)
        eval_model1 = eval(args.algorithm)(env, args)
        eval_model2 = eval(args.algorithm)(env, args)

        model = MultiAgent(env, [model1, model2], args, eval_models = [eval_model1, eval_model2], eval_env = eval_env)   
    
    else:
        model = MultiAgent(env, [model1, model2], args)

    ### Rollout
    rollout(env, model, args)

if __name__ == '__main__':
    parser.add_argument('--env', type=str, default=None, help='environment')
    parser.add_argument('--method', type=str, default=None, help='method name')
    parser_args = parser.parse_args()
    launch_rollout(parser_args.env, parser_args.method)