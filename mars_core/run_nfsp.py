from utils.func import LoadYAML2Dict
from env.import_env import make_env
from rollout import rollout
from rl.algorithm import *

### Load configurations
# yaml_file = 'confs/pettingzoo_boxingv1_nfsp'
# yaml_file = 'confs/slimevolley_slimevolleyv0_nfsp'
yaml_file = 'confs/gym_cartpolev1_nfsp'

args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)

### Create env
env = make_env(args)
print(env)

### Specify models for each agent
model1 = eval(args.algorithm)(env, args)
model2 = eval(args.algorithm)(env, args)
model2.fix()

model = MultiAgent(env, [model1, model2], args)

### Rollout
rollout(env, model, args)
