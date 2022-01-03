from mars.utils.func import LoadYAML2Dict
from mars.utils.common import EvaluationModelMethods
from mars.env.import_env import make_env
from mars.rollout import rollout
from mars.rl.agents import *
from mars.utils.func import get_general_args
import argparse
parser = argparse.ArgumentParser(description='Arguments of the general launching script for MARS.')

### Load configurations
game_type = 'pettingzoo'
game = ['boxing_v1', 'surround_v1', 'combat_plane_v1', \
        'combat_tank_v1', 'pong_v2', 'tennis_v2', \
        'ice_hockey_v1', 'double_dunk_v2'][0]

method = ['selfplay', 'selfplay2', 'fictitious_selfplay', \
            'fictitious_selfplay2', 'nash_dqn', 'nash_dqn_exploiter', \
            'nfsp', 'nash_ppo'][-1] 

# method = 'nash_dqn_speed'

args = get_general_args(game_type+'_'+game, method)
args.multiprocess = False
print(args)

### Create env
env = make_env(args)
print(env)

### Specify models for each agent     
model1 = eval(args.algorithm)(env, args)
model2 = eval(args.algorithm)(env, args)

if method in EvaluationModelMethods:
    args.eval_models = True
else:
    args.eval_models = False
model = MultiAgent(env, [model1, model2], args)

### Rollout
rollout(env, model, args)