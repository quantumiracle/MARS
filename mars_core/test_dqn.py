from env.import_env import make_env
from rl.dqn import DQN
from rl.agent import MultiAgent
from utils.data_struct import AttrDict

EnvArgs = {
    'name': 'pong_v1',
    'type': 'pettingzoo',
    'num_envs': 1, 
    'ram': True, 
    'against_baseline': False,
    'seed': 1223,
}

AgentArgs = {
    'hidden_dim': 64,
    'algorithm_spec': {'dueling': True},  # algorithm specific configurations
}

AgentArgs.update(EnvArgs)

def rollout(env, model):
    o = env.reset()
    max_frames = 10000
    for step in range(max_frames):
        a = model.choose_action(o)
        o_, r, d, _ = env.step(a)
        o = o_
        env.render()

        if d:
            break

if __name__ == "__main__":
    env_args = AttrDict(EnvArgs)
    test_env = make_env(env_args)

    args = AttrDict(AgentArgs)
    model1 = DQN(test_env, args)
    model2 = DQN(test_env, args)

    model = MultiAgent(test_env, [model1, model2])

    rollout(test_env, model)