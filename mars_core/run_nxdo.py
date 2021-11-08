from utils.func import LoadYAML2Dict
from env.import_env import make_env
from rollout import rollout
from rl.algorithm import *

### Load configurations
# yaml_file = 'confs/slimevolley_slimevolleyv0_nxdo_dqn'
yaml_file = 'confs/pettingzoo_boxingv1_nxdo_dqn'


args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)

### Create env
env = make_env(args)
print(env)

### Specify models for each agent
model1 = eval(args.algorithm)(env, args)
model2 = eval(args.algorithm)(env, args)
# model1.fix()  # fix a model if you don't want it to learn

args.num_envs = 1
eval_env = make_env(args)
eval_model1 = eval(args.algorithm)(env, args)
eval_model2 = eval(args.algorithm)(env, args)

model = MultiAgent(env, [model1, model2], args, eval_models = [eval_model1, eval_model2], eval_env = eval_env)

### Rollout
from datetime import datetime
now = datetime.now()
save_id = now.strftime("%Y%m%d%H%M%S")
rollout(env, model, args, save_id)

