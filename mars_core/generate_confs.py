### 
# This script generates configurations files for different training settings.
###
import os
import yaml, copy

two_player_zero_sum_games = ['combat_plane_v1', 'combat_tank_v1', 'surround_v1', 'space_war_v1', 'pong_v2', 'boxing_v1', 'tennis_v2', 'ice_hockey_v1']
methods = ['selfplay', 'selfplay2', 'fictitious_selfplay', 'fictitious_selfplay2', 'nfsp', 'nash_dqn', 'nash_dqn_exploiter', 'nash_ppo', 'nxdo', 'nxdo2']
game_type = 'pettingzoo'

self_play_method_marl_specs = {
        'selfplay_score_delta': 60,  # 10 the score that current learning agent must beat its opponent to update opponent's policy
        'trainable_agent_idx': 0,   # the index of trainable agent, with its opponent delayed updated
        'opponent_idx': 1   
        }

selfplay_based_methods = {'selfplay', 'selfplay2', 'fictitious_selfplay', 'fictitious_selfplay2', 'nxdo', 'nxdo2'}
large_nets_envs = {'surround_v1', 'ice_hockey_v1', 'combat_tank_v1'}

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
    'surround_v1': 16,
    'boxing_v1': 60,
    'combat_plane_v1': 10,
    'combat_tank_v1': 10,
    'space_war_v1': 10,
    'pong_v2': 20,
    'tennis_v2': 10,
    'ice_hockey_v1': 10,
}

train_start_frame = {  # for NFSP method only
    'slimevolley': 1000,
    'boxing_v1': 10000,
    'surround_v1': 10000,
    'combat_plane_v1': 10000,
    'combat_tank_v1': 10000,
    'space_war_v1': 10000,
    'pong_v2': 10000,
    'tennis_v2': 10000,
    'ice_hockey_v1': 10000,
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
    'policy':{
      'hidden_dim_list': [64, 64, 64, 64],
      'hidden_activation': 'ReLU',
      'output_activation': 'Softmax',
    },
    'value': {
      'hidden_dim_list': [64, 64, 64, 64],
      'hidden_activation': 'ReLU',
      'output_activation': False,
    }

}

times4_ppo_net_architecture = {
    'policy':{
      'hidden_dim_list': [256, 256, 256, 256],
      'hidden_activation': 'ReLU',
      'output_activation': 'Softmax',
    },
    'value': {
      'hidden_dim_list': [256, 256, 256, 256],
      'hidden_activation': 'ReLU',
      'output_activation': False,
    }

}

times4_net_architecture = {
    'hidden_dim_list': [256, 256, 256, 256],
    'hidden_activation': 'ReLU',
    'output_activation': False,
}


# creat folders for holding confs
for game in two_player_zero_sum_games:
    os.makedirs(f"confs/{game_type}/{game}", exist_ok=True)

# load general confs
with open(f'confs/{game_type}/{game_type}_general.yaml') as f:
    general_confs = yaml.safe_load(f)
    print(general_confs)

# dump env-task specific confs
for game in two_player_zero_sum_games:
    for method in methods:
        conf = copy.deepcopy(general_confs)
        conf['env_args']['env_name'] = game
        conf['train_args']['marl_method'] = method
        conf['train_args']['marl_spec'] = get_method_env_marl_spec(method, game)

        # some method specific confs
        if method in ['nash_dqn', 'nash_dqn_exploiter']:
            conf['train_args']['update_itr'] = 0.1
            conf['train_args']['marl_spec']['global_state'] = False
            if method == 'nash_dqn':
                conf['agent_args']['algorithm'] = 'NashDQN'
            elif method == 'nash_dqn_exploiter':
                conf['agent_args']['algorithm'] = 'NashDQNExploiter'
                conf['agent_args']['algorithm_spec']['exploiter_update_itr'] = 1

        elif method == 'nash_ppo':
            conf['train_args']['update_itr'] = 1
            conf['train_args']['marl_spec']['global_state'] = True
            conf['agent_args']['algorithm'] = 'NashPPO'
            conf['agent_args']['algorithm_spec'] = ppo_algorithm_spec
            conf['train_args']['net_architecture'] = ppo_net_architecture

        elif method == 'nfsp':
            conf['train_args']['train_start_frame'] = train_start_frame[game]

        # some game specific confs
        if game in large_nets_envs:  # it requires a larger net
            if method == 'nash_ppo':
                conf['train_args']['net_architecture'] = times4_ppo_net_architecture
            else:
                conf['train_args']['net_architecture'] = times4_net_architecture

        output_path = f"confs/{game_type}/{game}/{game_type}_{game}_{method}.yaml"
        with open(output_path, 'w') as outfile:
            yaml.dump(conf, outfile, default_flow_style=False, sort_keys=False)
            print(f'Dump confs: {output_path}.')