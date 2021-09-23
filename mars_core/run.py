from utils.func import LoadYAML2Dict
from env.import_env import make_env
from rollout import rollout
from rl.algorithm import *

import argparse
parser = argparse.ArgumentParser(
    description='NFSP training.')
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--model', type=str, default=None,
                    help='model index')
parser_args = parser.parse_args()

### Load configurations
# yaml_file = 'confs/pettingzoo_pongv1_selfplay_dqn'
# yaml_file = 'confs/pettingzoo_boxingv1_selfplay_dqn' 
# yaml_file = 'confs/pettingzoo_boxingv1_selfplay_ppo'
# yaml_file = 'confs/slimevolley_slimevolleyv0_selfplay_dqn'
# yaml_file = 'confs/slimevolley_slimevolleyv0_selfplay_ppo'
# yaml_file = 'confs/slimevolley_slimevolleyv0_fictitiousselfplay_dqn'
yaml_file = 'confs/pettingzoo_boxingv1_fictitiousselfplay_dqn'

args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)
if parser_args.test:
    args.test = True
    args.render = True
    args.load_model_full_path = '../model/pettingzoo_boxing_v1_fictitious_selfplay_DQN_20210921162239/9971'
    
### Create env
env = make_env(args)
print(env)

### Specify models for each agent
model1 = eval(args.algorithm)(env, args)
model2 = eval(args.algorithm)(env, args)
# model1.fix()  # fix a model if you don't want it to learn

model = MultiAgent(env, [model1, model2], args)

### Rollout
rollout(env, model, args)
