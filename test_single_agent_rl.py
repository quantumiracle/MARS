from mars.utils.func import LoadYAML2Dict
from mars.utils.wandb import init_wandb
from mars.env.import_env import make_env
from mars.rollout import rollout
from mars.rl.agents import *
from mars.rl.agents.multiagent import MultiAgent


# yaml_file = 'mars/confs/gym/cartpole_v1/gym_cartpole_v1_ppo' #PATH TO YAML
# yaml_file = 'mars/confs/gym/mountaincarcontinuous_v0/gym_mountaincarcontinuous_v0_ppo' #PATH TO YAML
# yaml_file = 'mars/confs/gym/ant_v2/gym_ant_v2_ppo' #PATH TO YAML
# yaml_file = 'mars/confs/gym/hopper_v2/gym_hopper_v2_ppo' #PATH TO YAML
yaml_file = 'mars/confs/gym/pong_v5/gym_pong_v5_dqn' #PATH TO YAML


args = LoadYAML2Dict(yaml_file, toAttr=True)
args.record_video = True
args.wandb_activate = True
args.wandb_project = 'pong-v5'
args.wandb_entity = 'quantumiracle'
# init_wandb(args)
### Create env
env = make_env(args)
print(env)

### Specify models for each agent
model = eval(args.algorithm)(env, args)

model = MultiAgent(env, [model], args)

### Rollout
rollout(env, model, args)