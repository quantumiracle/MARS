"""
This script is for testing the models trained in 
multi-agent setting with single-agent env (Gym),
not for testing single-agent models.
"""

from mars.utils.func import LoadYAML2Dict
from mars.env.import_env import make_env
from mars.rollout import rollout
from mars.rl.algorithm import *
from mars.utils.func import get_general_args, get_latest_file_in_folder
from mars.utils.common import SelfplayBasedMethods

def map_pettingzoo_to_gym(EnvNamePettingzoo):
    map_dict = {
        'boxing_v1': 'Boxing-ram-v0'
    }
    try:
        EnvNameGym = map_dict[EnvNamePettingzoo]
    except:
        print(f'No matched env in Gym for {EnvNamePettingzoo}.')
    print(f'From Pettingzoo env {EnvNamePettingzoo} to Gym env {EnvNameGym}.')
    return EnvNameGym


### Load configurations
game_type = 'pettingzoo'
game = ['boxing_v1', 'surround_v1', 'combat_plane_v1'][0]
method = ['selfplay', 'fictitious_selfplay', 'nash_dqn', 'nash_dqn_exploiter', 'nfsp', 'psro'][2]

args = get_general_args(game_type+'_'+game, method)
print(args)

## Change/specify some arguments if necessary
args.max_episodes = 1000
args.against_baseline = False
args.test = True
args.exploit = False
args.render = True
folder = f'../data/model/20211120_1257/{game_type}_{game}_{method}/'
if method in SelfplayBasedMethods:
    file_path = get_latest_file_in_folder(folder)
else:
    file_path = get_latest_file_in_folder(folder, id=0)  # load from the first agent model of the two
args.load_model_full_path = file_path

# Change to test args in single-agent env
args.env_name = map_pettingzoo_to_gym(game)
args.env_type = 'gym'


### Create env
env = make_env(args)

### Specify models for each agent
model1 = eval(args.algorithm)(env, args)

model = MultiAgent(env, [model1], args)

### Rollout
from datetime import datetime
now = datetime.now()
save_id = now.strftime("%Y%m%d%H%M%S")
rollout(env, model, args, save_id)

