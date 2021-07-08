from utils.func import LoadYAML2Dict
from env.import_env import make_env
from rollout_ga import rollout_ga
from rl.algorithm import *

### Load configurations
yaml_file = 'confs/gym_cartpolev1_ga'

args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)

### Create env
env = make_env(args)
print(env)

### Specify models for each agent
model = eval(args.algorithm)(env, args)

### Rollout
rollout_ga(env, model, args)

