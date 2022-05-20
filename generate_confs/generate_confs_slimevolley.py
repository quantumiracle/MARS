### 
# This script generates configurations files for different training settings.
###
import os
import yaml, copy

target_path = '../'  # from the root of the MARS

games = ['SlimeVolley-v0']

methods = ['selfplay', 'selfplay2', 'fictitious_selfplay', \
            'fictitious_selfplay2', 'nfsp', 'nash_dqn', \
            'nash_dqn_exploiter', 'nash_dqn_factorized', 'nash_ppo', 'psro_sym', 'psro']

game_type = 'slimevolley'

self_play_method_marl_specs = {
        'selfplay_score_delta': 4,  # 10 the score that current learning agent must beat its opponent to update opponent's policy
        'trainable_agent_idx': 0,   # the index of trainable agent, with its opponent delayed updated
        'opponent_idx': 1   
        }

selfplay_based_methods = {'selfplay', 'selfplay2', 'fictitious_selfplay', \
                            'fictitious_selfplay2', 'psro_sym', 'psro'}

def get_method_env_marl_spec(method, env):
    if method in selfplay_based_methods:
        self_play_method_marl_specs_ = copy.deepcopy(self_play_method_marl_specs)
        self_play_method_marl_specs_['selfplay_score_delta'] = selfplay_score_deltas[env]
        marl_spec = self_play_method_marl_specs_

    elif method == 'nfsp':
        marl_spec =  {
        'eta': 0.1
    }

    else:
        marl_spec = {}

    return marl_spec


selfplay_score_deltas = { # specific for each environment
    'SlimeVolley-v0': 3,
}

train_start_frame = {  # for NFSP method only
    'SlimeVolley-v0': 10000,
}


ppo_algorithm_spec = { # specs for PPO alg.
    'episodic_update': True,  # as PPO is on-policy, it uses episodic update instead of update per timestep
    'gamma': 0.99,
    'lambda': 0.95,
    'eps_clip': 0.2,
    'K_epoch': 4,
    'GAE': True,
}

ppo_net_architecture = {
    'feature':{
      'hidden_dim_list': [128, 128],
      'hidden_activation': 'ReLU',
      'output_activation': False,
    },
    'policy':{
      'hidden_dim_list': [128],
      'hidden_activation': False,
      'output_activation': 'Softmax',
    },
    'value': {
      'hidden_dim_list': [128],
      'hidden_activation': 'ReLU',
      'output_activation': False,
    }

}

standard_net_architecture = {
    'hidden_dim_list': [128, 128, 128],
    'hidden_activation': 'ReLU',
    'output_activation': False,
}


# creat folders for holding confs
for game in games:
    os.makedirs(target_path+f"mars/confs/{game_type}/{game}", exist_ok=True)

# load general confs
with open(target_path+f'mars/confs/{game_type}/{game_type}_general.yaml') as f:
    general_confs = yaml.safe_load(f)
    print(general_confs)

# dump env-task specific confs
for game in games:
    for method in methods:
        conf = copy.deepcopy(general_confs)
        conf['env_args']['env_name'] = game
        conf['train_args']['marl_method'] = method
        conf['train_args']['marl_spec'] = get_method_env_marl_spec(method, game)

        conf['env_args']['num_envs'] = 5
        conf['train_args']['max_episodes'] = 10000
        conf['train_args']['max_steps_per_episode'] = 300 # truncated game for speed up
        conf['agent_args']['algorithm_spec']['eps_decay'] = 10*conf['train_args']['max_episodes']  # proper for training 10000 episodes
        conf['agent_args']['algorithm_spec']['multi_step'] = 1

        # some method specific confs
        if method in ['nash_dqn', 'nash_dqn_exploiter', 'nash_dqn_factorized']:
            conf['env_args']['num_envs'] = 1
            conf['train_args']['max_episodes'] = 50000
            conf['agent_args']['algorithm_spec']['eps_decay'] = 100*conf['train_args']['max_episodes']  # proper for training 10000 episodes
            conf['train_args']['update_itr'] = 1  # 0.1
            conf['train_args']['marl_spec']['global_state'] = False
            if method == 'nash_dqn':
                conf['agent_args']['algorithm'] = 'NashDQN'
            if method == 'nash_dqn_factorized':
                conf['agent_args']['algorithm'] = 'NashDQNFactorized'
            elif method == 'nash_dqn_exploiter':
                conf['agent_args']['algorithm'] = 'NashDQNExploiter'
                conf['agent_args']['algorithm_spec']['exploiter_update_itr'] = 1

        elif method == 'nash_ppo':
            conf['train_args']['multiprocess'] = False
            conf['train_args']['update_itr'] = 1
            conf['train_args']['marl_spec']['global_state'] = True
            conf['agent_args']['algorithm'] = 'NashPPO'
            conf['agent_args']['algorithm_spec'] = ppo_algorithm_spec
            conf['train_args']['net_architecture'] = ppo_net_architecture

        elif method == 'nfsp':
            conf['agent_args']['algorithm'] = 'NFSP'
            conf['train_args']['net_architecture']['policy'] = standard_net_architecture
            conf['train_args']['net_architecture']['policy']['output_activation'] = 'Softmax'
            conf['train_args']['train_start_frame'] = train_start_frame[game]

        output_path = target_path+f"mars/confs/{game_type}/{game}/{game_type}_{game}_{method}.yaml"
        with open(output_path, 'w') as outfile:
            yaml.dump(conf, outfile, default_flow_style=False, sort_keys=False)
            print(f'Dump confs: {output_path}.')