from utils.func import LoadYAML2Dict
from env.import_env import make_env
from rollout import rollout
from rl.algorithm import *
import argparse
parser = argparse.ArgumentParser(
    description='Imitation learning training.')
parser.add_argument('--id', dest='id', action='store', default=0)
parser_args = parser.parse_args()

### Load configurations
yaml_file = 'confs/pettingzoo_atari_nash_dqn_exploiter'
args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)
envs = ['basketball_pong_v2', 'surround_v1', 'boxing_v1']
args.env_name = envs[int(parser_args.id)]

### Create env
env = make_env(args)
print(env)

### Specify models for each agent
model1 = eval(args.algorithm)(env, args)

model = MultiAgent(env, [model1], args)

### Rollout
rollout(env, model, args)
