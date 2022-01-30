from mars.utils.func import LoadYAML2Dict
from mars.rollout import rollout
from mars.rl.agents import *
from mars.rl.agents.multiagent import MultiAgent
from mars.env.import_env import make_env

import argparse
parser = argparse.ArgumentParser(
    description='Training configurations.')
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--model', type=str, default=None,
                    help='model index')
parser_args = parser.parse_args()

### Load configurations

method = ['fictitious_selfplay2', 'nash_dqn', 'nash_dqn_exploiter', 'nxdo2', 'nfsp', 'nash_ppo'][-1]
prefix = 'mars/confs/mdp/mdp_arbitrary_mdp_'
yaml_file = prefix + method

args = LoadYAML2Dict(yaml_file, toAttr=True)
args.marl_spec['global_state'] = True
args.max_episodes = 50100
if parser_args.test:
    args.test = True
    args.render = True
    args.load_model_idx = parser_args.model
args.device = 'gpu'
### Create env
env = make_env(args)
print(env)

### Specify models for each agent
model1 = eval(args.algorithm)(env, args)
model2 = eval(args.algorithm)(env, args)

model = MultiAgent(env, [model1, model2], args)

### Rollout
from datetime import datetime
now = datetime.now()
save_id = now.strftime("%Y%m%d%H%M%S")
rollout(env, model, args, save_id)

