from mars.env.import_env import make_env
from mars.rollout import rollout
from mars.rl.agents import *
from mars.rl.agents.multiagent import MultiAgent
from mars.utils.func import get_general_args
import argparse
parser = argparse.ArgumentParser(description='Arguments of the general launching script for MARS.')

### Load configurations
game_type = 'slimevolley'
game = ['SlimeVolley-v0'][0]

method = ['selfplay', 'selfplay2', 'fictitious_selfplay', \
            'fictitious_selfplay2', 'nash_dqn', 'nash_dqn_exploiter', \
            'nfsp', 'nxdo2', 'nash_ppo'][0] 

# method = 'nash_dqn_speed'

args = get_general_args(game_type+'_'+game, method)
args.num_envs = 1
args.multiprocess = False
print(args)

### Create env
env = make_env(args)
print(env)

### Specify models for each agent     
model1 = eval(args.algorithm)(env, args)
model2 = eval(args.algorithm)(env, args)

model = MultiAgent(env, [model1, model2], args)

### Rollout
rollout(env, model, args)