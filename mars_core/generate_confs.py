### 
# This script generates configurations files for different training settings.
###
import os
import yaml, copy

two_player_zero_sum_games = ['combat_plane_v1', 'combat_tank_v1', 'surround_v1', 'space_war_v1', 'pong_v1', 'boxing_v1']
methods = ['selfplay', 'fictitious_selfplay',  'nfsp', 'nash_dqn', 'nash_dqn_exploiter', 'nxdo', 'nxdo2']
game_type = 'pettingzoo'

self_play_method_marl_specs = {
        'selfplay_score_delta': 60,  # 10 the score that current learning agent must beat its opponent to update opponent's policy
        'trainable_agent_idx': 0,   # the index of trainable agent, with its opponent delayed updated
        'opponent_idx': 1   
        }

selfplay_based_methods = {'selfplay', 'fictitious_selfplay', 'nxdo', 'nxdo2'}

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
    'pong_v1': 20,
}

train_start_frame = {  # for NFSP method only
    'slimevolley': 1000,
    'boxing_v1': 10000,
    'surround_v1': 10000,
    'combat_plane_v1': 10000,
    'combat_tank_v1': 10000,
    'space_war_v1': 10000,
    'pong_v1': 10000,
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
            if method == 'nash_dqn':
                conf['agent_args']['algorithm'] = 'NashDQN'
            elif method == 'nash_dqn_exploiter':
                conf['agent_args']['algorithm'] = 'NashDQNExploiter'
                conf['agent_args']['algorithm_spec']['exploiter_update_itr'] = 1

        elif method == 'nfsp':
            conf['train_args']['train_start_frame'] = train_start_frame[game]

        output_path = f"confs/{game_type}/{game}/{game_type}_{game}_{method}.yaml"
        with open(output_path, 'w') as outfile:
            yaml.dump(conf, outfile, default_flow_style=False, sort_keys=False)
            print(f'Dump confs: {output_path}.')