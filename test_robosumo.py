import numpy as np

from mars.utils.func import LoadYAML2Dict
from mars.env.import_env import make_env
from mars.rollout import rollout
from mars.rl.agents import *
from mars.rl.agents.multiagent import MultiAgent

game_type = ['robosumo'][0]

game = ['RoboSumo-Ant-vs-Ant-v0'][0]

# game_name = game.lower().split('-')[1:]
# game = '_'.join(game_name)

yaml_file = f'mars/confs/{game_type}/{game}/{game_type}_{game}_nash_ppo' #PATH TO YAML

args = LoadYAML2Dict(yaml_file, toAttr=True)

### Create env
# args.render = True
args.record_video = True
env = make_env(args)
print(env)

### Test env
obser = env.reset()
skip = 0
for t in range(1000):
    actions = [a_space.sample() for a_space in env.action_spaces.values()]
    print(t, actions)
    obser, r, done, info = env.step(actions)
    env.render(mode='rgb_array')
    # env.render()
    if np.any(done): break

### Specify models for each agent
model1 = eval(args.algorithm)(env, args)
model2 = eval(args.algorithm)(env, args)

model = MultiAgent(env, [model1, model2], args)

### Rollout
rollout(env, model, args)