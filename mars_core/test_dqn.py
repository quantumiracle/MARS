from env.import_env import make_env
from rl.dqn import DQN
from rl.agent import MultiAgent
from utils.data_struct import AttrDict

EnvArgs = {
    'env_name': 'pong_v1',
    'env_type': 'pettingzoo',
    'num_envs': 1, 
    'ram': True, 
    'against_baseline': False,
    'seed': 1223,
}

AgentArgs = {
    'hidden_dim': 64,
    'algorithm_spec':  # algorithm specific configurations
        {'dueling': True,
        'replay_buffer_size': 1e5,
        'gamma': 0.99,
        'multi_step': 1,
        'target_update_interval': 1000,
        'eps_start': 1.,
        'eps_final': 0.01,
        'eps_decay': 30000
        },  
}

TrainArgs = {
    'batch_size': 64,
    'max_episodes': 10000,
    'max_steps_per_episode': 10000,
    'optimizer': 'adam',
    'learning_rate': 1e-4,
    'device': 'gpu',
    'update_itr': 1,
    'log_avg_window': 5,
    'log_interval': 20,
    'render': False
}

AgentArgs.update(TrainArgs)

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

    model = MultiAgent(test_env, [model1, model2], args)

    rollout(test_env, model)