from mars.utils.func import LoadYAML2Dict
from mars.env.import_env import make_env
from mars.rollout import rollout
from mars.rl.agents import *
from mars.rl.agents.multiagent import MultiAgent

game_type = ['safetygym'][0]

game = ['Safexp-PointGoal1-v0'][0]

yaml_file = f'mars/confs/{game_type}/{game}/{game_type}_{game}_dqn' #PATH TO YAML

args = LoadYAML2Dict(yaml_file, toAttr=True)

### Create env
env = make_env(args)
print(env)

### Specify models for each agent
model = eval(args.algorithm)(env, args)

model = MultiAgent(env, [model], args)

### Rollout
rollout(env, model, args)