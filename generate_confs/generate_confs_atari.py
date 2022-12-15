### 
# This script generates configurations files for different training settings.
###
import os
import yaml, copy

target_path = '../'  # from the root of the MARS

two_player_zero_sum_games = ['combat_jet_v1', 'combat_tank_v2', 'surround_v2', \
                            'space_war_v2', 'pong_v3', 'basketball_pong_v3', 'boxing_v2', \
                            'tennis_v3', 'ice_hockey_v2', 'double_dunk_v3']

methods = ['selfplay', 'selfplay_sym', 'fictitious_selfplay', \
            'fictitious_selfplay_sym', 'nfsp', 'nash_dqn', \
            'nash_dqn_exploiter', 'nash_dqn_factorized', 'nash_ppo', 'psro_sym', 'psro']

game_type = 'pettingzoo'

self_play_method_marl_specs = {
        'selfplay_score_delta': 60,  # 10 the score that current learning agent must beat its opponent to update opponent's policy
        'trainable_agent_idx': 0,   # the index of trainable agent, with its opponent delayed updated
        'opponent_idx': 1   
        }

selfplay_based_methods = {'selfplay', 'selfplay_sym', 'fictitious_selfplay', \
                            'fictitious_selfplay_sym', 'psro_sym', 'psro'}

# large_nets_envs = {'tennis_v2'}
large_nets_envs = {}

ram = False

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


# for full episode length
# selfplay_score_deltas = { # specific for each environment
#     'surround_v2': 16,
#     'boxing_v2': 80,
#     'combat_jet_v1': 10, # this need to be tuned
#     'combat_tank_v2': 10,  # this need to be tuned
#     'space_war_v2': 10,
#     'pong_v3': 30,
#     'basketball_pong_v3': 30,
#     'tennis_v3': 50,
#     'ice_hockey_v2': 10,
#     'double_dunk_v3': 50,
# }

# for truncated games with 300 episode length
selfplay_score_deltas = { # specific for each environment
    'surround_v2': 3,
    'boxing_v2': 80,
    'combat_jet_v1': 5, # this need to be tuned
    'combat_tank_v2': 5,  # this need to be tuned
    'space_war_v2': 3,
    'pong_v3': 10,
    'basketball_pong_v3': 5,
    'tennis_v3': 7,
    'ice_hockey_v2': 2,
    'double_dunk_v3': 15,
}

train_start_frame = {  # for NFSP method only
    'slimevolley': 1000,
    'boxing_v2': 10000,
    'surround_v2': 10000,
    'combat_jet_v1': 10000,
    'combat_tank_v2': 10000,
    'space_war_v2': 10000,
    'pong_v3': 10000,
    'basketball_pong_v3': 10000,
    'tennis_v3': 10000,
    'ice_hockey_v2': 10000,
    'double_dunk_v3': 10000,
}


ppo_algorithm_spec = { # specs for PPO alg.
    'episodic_update': True,  # as PPO is on-policy, it uses episodic update instead of update per timestep
    'gamma': 0.99,
    'lambda': 0.95,
    'eps_clip': 0.2,
    'K_epoch': 4,
    'GAE': True,
    'max_grad_norm': 0.5,
    'entropy_coeff': 0.01,
    'vf_coeff': 0.5,
    'policy_loss_coeff': 0.08,

}

ppo_net_architecture = {
    'feature':{
      'hidden_dim_list': [128, 128],
      'hidden_activation': 'ReLU',
      'output_activation': False,
    },
    'policy':{
      'hidden_dim_list': [128, 128],
      'hidden_activation': False,
      'output_activation': 'Softmax',
    },
    'value': {
      'hidden_dim_list': [128, 128],
      'hidden_activation': 'ReLU',
      'output_activation': False,
    }

}

large_ppo_net_architecture = {
    'feature':{
      'hidden_dim_list': [256, 256],
      'hidden_activation': 'ReLU',
      'output_activation': False,
    },
    'policy':{
      'hidden_dim_list': [256],
      'hidden_activation': False,
      'output_activation': 'Softmax',
    },
    'value': {
      'hidden_dim_list': [256],
      'hidden_activation': 'ReLU',
      'output_activation': False,
    }

}

cnn_ppo_net_architecture = {
    'feature':{
    'hidden_dim_list': [512,],
    'channel_list': [32, 64, 64],
    'kernel_size_list': [8, 4, 3],
    'stride_list': [4, 2, 1],
      'hidden_activation': 'ReLU',
      'output_activation': False,
    },
    'policy':{
      'hidden_dim_list': [512,],
      'hidden_activation': False,
      'output_activation': 'Softmax',
    },
    'value': {
      'hidden_dim_list': [512,],
      'hidden_activation': 'ReLU',
      'output_activation': False,
    }

}

standard_net_architecture = {
    'hidden_dim_list': [128, 128, 128, 128],
    'hidden_activation': 'ReLU',
    'output_activation': False,
}

large_net_architecture = {
    'hidden_dim_list': [256, 256, 256, 256],
    'hidden_activation': 'ReLU',
    'output_activation': False,
}

cnn_net_architecture = {
    'hidden_dim_list': [512, 512],
    'channel_list': [32, 64, 64],
    'kernel_size_list': [8, 4, 3],
    'stride_list': [4, 2, 1],
    'hidden_activation': 'ReLU',
    'output_activation': False,
}




# creat folders for holding confs
for game in two_player_zero_sum_games:
    os.makedirs(target_path+f"mars/confs/{game_type}/{game}", exist_ok=True)

# load general confs
with open(target_path+f'mars/confs/{game_type}/{game_type}_general.yaml') as f:
    general_confs = yaml.safe_load(f)
    print(general_confs)

# dump env-task specific confs
for game in two_player_zero_sum_games:
    for method in methods:
        conf = copy.deepcopy(general_confs)
        conf['env_args']['env_name'] = game
        conf['train_args']['marl_method'] = method
        conf['train_args']['marl_spec'] = get_method_env_marl_spec(method, game)

        conf['env_args']['num_envs'] = 5
        conf['train_args']['max_episodes'] = 10000

        conf['train_args']['max_steps_per_episode'] = 300 # 300 truncated game for speed up
        conf['agent_args']['algorithm_spec']['eps_decay'] = 10*conf['train_args']['max_episodes']  # decay faster
        conf['agent_args']['algorithm_spec']['multi_step'] = 1

        # image-based input
        if not ram:
            conf['env_args']['ram'] = False
            conf['train_args']['net_architecture'] = copy.deepcopy(cnn_net_architecture)  # copy to make original not changed        
        print(game, method, conf['train_args']['net_architecture'])
        # some game specific confs
        if game in large_nets_envs:  # it requires a larger net
            conf['train_args']['net_architecture'] = copy.deepcopy(large_net_architecture)

        # some method specific confs
        if method in ['nash_dqn', 'nash_dqn_exploiter', 'nash_dqn_factorized']:
            # conf['env_args']['num_envs'] = 1
            # conf['train_args']['max_episodes'] = 50000
            conf['train_args']['max_episodes'] = 10000
            conf['agent_args']['algorithm_spec']['eps_decay'] = 100*conf['train_args']['max_episodes'] # 1000000  # proper for training 10000 episodes
            conf['train_args']['update_itr'] = 1
            conf['train_args']['marl_spec']['global_state'] = False
            if method == 'nash_dqn':
                conf['agent_args']['algorithm'] = 'NashDQN'
            if method == 'nash_dqn_factorized':
                conf['agent_args']['algorithm'] = 'NashDQNFactorized'
            elif method == 'nash_dqn_exploiter':
                conf['agent_args']['algorithm'] = 'NashDQNExploiter'
                conf['agent_args']['algorithm_spec']['exploiter_update_itr'] = 3

        elif method == 'nash_ppo':
            conf['train_args']['multiprocess'] = False
            conf['train_args']['update_itr'] = 1
            conf['train_args']['marl_spec']['global_state'] = True
            conf['agent_args']['algorithm'] = 'NashPPO'
            conf['agent_args']['algorithm_spec'] = ppo_algorithm_spec
            if not ram:
                conf['train_args']['net_architecture'] = cnn_ppo_net_architecture
            else:
                conf['train_args']['net_architecture'] = large_ppo_net_architecture if game in large_nets_envs else ppo_net_architecture

        elif method == 'nfsp':
            conf['agent_args']['algorithm'] = 'NFSP'
            if not ram:
                conf['train_args']['net_architecture']['policy'] = copy.deepcopy(cnn_net_architecture)
            else:
                if game in large_nets_envs:
                    conf['train_args']['net_architecture']['policy'] = large_net_architecture
                else:
                    conf['train_args']['net_architecture']['policy'] = standard_net_architecture
            conf['train_args']['net_architecture']['policy']['output_activation'] = 'Softmax'
            conf['train_args']['train_start_frame'] = train_start_frame[game]

        output_path = target_path+f"mars/confs/{game_type}/{game}/{game_type}_{game}_{method}.yaml"
        with open(output_path, 'w') as outfile:
            yaml.dump(conf, outfile, default_flow_style=False, sort_keys=False)
            print(f'Dump confs: {output_path}.')

        del conf
print(general_confs)