from utils.func import LoadYAML2Dict
from env.import_env import make_env
from rollout import rollout
from rl.algorithm import *

### Load configurations
# yaml_file = 'confs/gym_cartpolev1_ga'
yaml_file = 'confs/pettingzoo_boxingv1_selfplay_ga'
# yaml_file = 'confs/slimevolley_slimevolleyv0_selfplay_ga'

args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)

### Create env
env = make_env(args)
print(env)

### Specify models for each agent
model = eval(args.algorithm)(env, args)

### Rollout
rollout(env, model, args)

