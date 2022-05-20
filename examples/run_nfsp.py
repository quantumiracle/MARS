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
# yaml_file = 'confs/pettingzoo_boxingv1_nfsp'
yaml_file = 'confs/slimevolley_slimevolleyv0_nfsp'
# yaml_file = 'confs/gym_cartpolev1_nfsp'
# yaml_file = 'confs/lasertag_LaserTagsmallv0_nfsp'

args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)
if parser_args.test:
    args.test = True
    args.render = True
    args.load_model_idx = parser_args.model
    args.eta = 0.

### Create env
env = make_env(args)
print(env)

### Specify models for each agent
model1 = eval(args.algorithm)(env, args)
model2 = eval(args.algorithm)(env, args)

model = MultiAgent(env, [model1, model2], args)

### Rollout
from datetime import datetime
now = datetime.now()
save_id = now.strftime("%Y%m%d%H%M%S")
rollout(env, model, args, save_id)

