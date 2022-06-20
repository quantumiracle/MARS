from mars.utils.func import LoadYAML2Dict
from mars.env.import_env import make_env
from mars.rollout import rollout
from mars.rl.agents import *
from mars.rl.agents.multiagent import MultiAgent


yaml_file = 'mars/confs/gym_cartpolev1_ppo' #PATH TO YAML
# yaml_file = 'mars/confs/gym_mountaincarcontinuousv0_ppo' #PATH TO YAML
# yaml_file = 'mars/confs/gym_carracingv1_ppo' #PATH TO YAML

args = LoadYAML2Dict(yaml_file, toAttr=True)

### Create env
args.render = True
env = make_env(args)
print(env)

### Specify models for each agent
model = eval(args.algorithm)(env, args)

model = MultiAgent(env, [model], args)

### Rollout
rollout(env, model, args)