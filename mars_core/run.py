from utils.func import LoadYAML2Dict
from env.import_env import make_env
from rollout import rollout
from rl.dqn import DQN
from rl.agent import MultiAgent

### Load configurations
yaml_file = 'confs/pettingzoo_pongv1_selfplay'
args = LoadYAML2Dict(yaml_file, toAttr=True)
print(args)


### Create env
env = make_env(args)
print(env)

### Specify models for each agent
model1 = DQN(env, args)
model2 = DQN(env, args)

model = MultiAgent(env, [model1, model2])

### Rollout
rollout(env, model, args)