from utils.func import LoadYAML2Dict
from env.import_env import make_env
from rollout import rollout
from rl.algorithm import *

### Load configurations
# yaml_file = 'confs/gym_pongramv0_dqn'
# yaml_file = 'confs/gym_boxingramv0_dqn'
# yaml_file = 'confs/gym_icehockeyramv0_dqn'
yaml_file = 'confs/gym_icehockeyramv0_ppo'
# yaml_file = 'confs/gym_cartpolev1_dqn'
# yaml_file = 'confs/gym_cartpolev1_ppo'
# yaml_file = 'confs/slimevolley_slimevolleyv0_dqn'
# yaml_file = 'confs/slimevolley_slimevolleyv0_ppo'

args = LoadYAML2Dict(yaml_file, toAttr=True)

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

