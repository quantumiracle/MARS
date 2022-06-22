from mars.env.import_env import make_env
from mars.rollout import rollout
from mars.rl.agents import *
from mars.rl.agents.multiagent import MultiAgent
from mars.utils.func import get_general_args
import argparse
parser = argparse.ArgumentParser(description='Arguments of the general launching script for MARS.')

### Load configurations
game_type = ['pettingzoo', 'slimevolley'][0]

game = ['boxing_v2', 'surround_v2', 'combat_jet_v1', \
        'combat_tank_v2', 'pong_v3', 'tennis_v3', \
        'ice_hockey_v2', 'double_dunk_v3', 'SlimeVolley-v0'][4]

method = ['selfplay', 'selfplay_sym', 'fictitious_selfplay', \
            'fictitious_selfplay_sym', 'nash_dqn', 'nash_dqn_exploiter', \
            'nash_dqn_factorized', 'nfsp', 'psro', 'psro_sym', 'nash_ppo'][4] 

args = get_general_args(game_type+'_'+game, method)
#args.multiprocess = False
args.num_envs = 16
# args.device = 'cpu'
print(args)
args.render=False

### Create env
env = make_env(args)
print(env)

### Specify models for each agent     
model1 = eval(args.algorithm)(env, args)
model2 = eval(args.algorithm)(env, args)
print(model1, model2)

model = MultiAgent(env, [model1, model2], args)

### Rollout
rollout(env, model, args)
