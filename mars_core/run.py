from utils.func import LoadYAML2Dict
from env.import_env import make_env
from rollout import rollout
from rl.algorithm import *

### Load configurations
# yaml_file = 'confs/pettingzoo_pongv1_selfplay'
# yaml_file = 'confs/pettingzoo_boxingv1_dqn'
# yaml_file = 'confs/pettingzoo_boxingv1_selfplay'
yaml_file = 'confs/slimevolley_slimevolleyv0_selfplay'
# yaml_file = 'confs/slimevolley_slimevolleyv0_dqn'
# yaml_file = 'confs/slimevolley_slimevolleyv0_ppo'

args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)

### Create env
env = make_env(args)
print(env)

### Specify models for each agent
model1 = eval(args.algorithm)(env, args)
model2 = eval(args.algorithm)(env, args)
# model1.fix()  # fix model1

model = MultiAgent(env, [model1, model2], args)

### Rollout
rollout(env, model, args)