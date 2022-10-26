import numpy as np

from mars.utils.func import LoadYAML2Dict
from mars.env.import_env import make_env
from mars.rollout import rollout
from mars.rl.agents import *
from mars.rl.agents.multiagent import MultiAgent

game_type = 'advmujoco'

game = ['ant_v2'][0]

# yaml_file = f'mars/confs/{game_type}/{game}/{game_type}_{game}_nash_ppo' #PATH TO YAML
yaml_file = f'mars/confs/{game_type}/{game}/{game_type}_{game}_nash_actor_critic' #PATH TO YAML

args = LoadYAML2Dict(yaml_file, toAttr=True)

### Create env
# args.render = True
env = make_env(args)
print(env, env.action_spaces)

### Specify models for each agent
model1 = eval(args.algorithm)(env, args)
model2 = eval(args.algorithm)(env, args)

model = MultiAgent(env, [model1, model2], args)

### Rollout
rollout(env, model, args)