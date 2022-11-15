### Test environment loading ###

import sys
sys.path.append("..")
from mars.env.import_env import make_env
from mars.utils.data_struct import AttrDict

EnvArgs = {
    'name': None,
    'type': None,
    'num_envs': 2, 
    'ram': True, 
    'against_baseline': False,
    'seed': 1223,
    }

# names of environments to test
envs = {
    'slimevolley': [
        'SlimeVolley-v0', 'SlimeVolleySurvivalNoFrameskip-v0',
        'SlimeVolleyNoFrameskip-v0', 'SlimeVolleyPixel-v0'
    ],
    'pettingzoo': [
        'basketball_pong_v3', 'boxing_v2', 'combat_plane_v2', 'combat_tank_v2',
        'double_dunk_v3', 'entombed_competitive_v3', 'entombed_cooperative_v3',
        'flag_capture_v2', 'foozpong_v3', 'ice_hockey_v2', 'joust_v3',
        'mario_bros_v3', 'maze_craze_v3', 'othello_v3', 'pong_v3',
        'quadrapong_v4', 'space_invaders_v2', 'space_war_v2', 'surround_v2',
        'tennis_v3', 'video_checkers_v4', 'volleyball_pong_v3', 'warlords_v3',
        'wizard_of_wor_v3',
        'dou_dizhu_v4', 'go_v5', 'leduc_holdem_v4', 'rps_v2',
        'texas_holdem_no_limit_v6', 'texas_holdem_v4', 'tictactoe_v3', 'uno_v4'
    ],
    # 'lasertag':
    # ['LaserTag-small2-v0', 'LaserTag-small3-v0', 'LaserTag-small4-v0'],
    'gym':
    ['ALE/Pong-v5', 'LunarLander-v2', 'CartPole-v1' ]  # 'HalfCheetah-v2' requires MuJoCo and mujoco-py to be installed
}


if __name__ == "__main__":
    cnt, fail_cnt = 0, 0
    for env_type, envs in envs.items():
        for env_name in envs:
            cnt += 1
            EnvArgs['env_name'] = env_name
            EnvArgs['env_type'] = env_type
            EnvArgs['num_process'] = 1
            EnvArgs['record_video'] = False
            env_args = AttrDict(EnvArgs)
            test_env = make_env(env_args, ss_vec=False)

            try:
                test_env = make_env(env_args)
            except:
                fail_cnt += 1
                print(f'Failed to load env {env_name} in type {env_type}.')
            print(f"{cnt-fail_cnt}/{cnt} succeed.")
