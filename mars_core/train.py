from utils.func import LoadYAML2Dict
from utils.common import EvaluationModelMethods
from env.import_env import make_env
from rollout import rollout
from rl.algorithm import *
from general_train import get_general_args
import argparse
parser = argparse.ArgumentParser(description='Arguments of the general launching script for MARS.')

### Load configurations
game_type = 'pettingzoo'
game = ['boxing_v1', 'surround_v1', 'combat_plane_v1', 'pong_v2', 'tennis_v2', 'ice_hockey_v1'][-1]
method = ['selfplay', 'selfplay2', 'fictitious_selfplay', 'fictitious_selfplay2', 'nash_dqn', 'nash_dqn_exploiter', 'nash_ppo'][-1]

args = get_general_args(game_type+'_'+game, method)
print(args)

### Create env
env = make_env(args)
print(env)

### Specify models for each agent     
model1 = eval(args.algorithm)(env, args)
model2 = eval(args.algorithm)(env, args)

if method in EvaluationModelMethods:
    eval_env = make_env(args)
    eval_model1 = eval(args.algorithm)(env, args)
    eval_model2 = eval(args.algorithm)(env, args)

    model = MultiAgent(env, [model1, model2], args, eval_models = [eval_model1, eval_model2], eval_env = eval_env)   

else:
    model = MultiAgent(env, [model1, model2], args)

### Rollout
rollout(env, model, args)