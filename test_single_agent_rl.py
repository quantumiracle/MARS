from mars.utils.func import LoadYAML2Dict
from mars.env.import_env import make_env
from mars.rollout import rollout
from mars.rl.agents import *
from mars.rl.agents.multiagent import MultiAgent


# yaml_file = 'mars/confs/gym/cartpole_v1/gym_cartpole_v1_ppo' #PATH TO YAML
# yaml_file = 'mars/confs/gym/mountaincarcontinuous_v0/gym_mountaincarcontinuous_v0_ppo' #PATH TO YAML
# yaml_file = 'mars/confs/gym/ant_v2/gym_ant_v2_ppo' #PATH TO YAML
yaml_file = 'mars/confs/gym/hopper_v2/gym_hopper_v2_ppo' #PATH TO YAML

args = LoadYAML2Dict(yaml_file, toAttr=True)
### Create env
env = make_env(args)
print(env)

### Specify models for each agent
model = eval(args.algorithm)(env, args)

model = MultiAgent(env, [model], args)

### Rollout
rollout(env, model, args)