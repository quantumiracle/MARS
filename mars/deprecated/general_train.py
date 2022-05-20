from utils.func import LoadYAML2Dict
from env.import_env import make_env
from rollout import rollout
from rl.algorithm import *
import argparse
parser = argparse.ArgumentParser(description='Arguments of the general launching script for MARS.')

def get_general_args(env, method):
    prefix = 'confs/'
    if 'nash' in method:
        postfix = ''
    else:
        postfix = '_dqn'
    yaml_file = prefix+env+'_'+method+postfix
    args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)
    return args
    
def launch_rollout(env, method, save_id):
    args = get_general_args(env, method)

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
    rollout(env, model, save_id, args)

if __name__ == '__main__':
    parser.add_argument('--env', type=str, default=None, help='environment')
    parser.add_argument('--method', type=str, default=None, help='method name')
    parser.add_argument('--save_id', type=str, default=None, help='identification number for each run')
    parser_args = parser.parse_args()
    launch_rollout(parser_args.env, parser_args.method, parser_args.save_id)