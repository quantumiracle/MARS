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
    # 'slimevolley': [
    #     'SlimeVolley-v0', 'SlimeVolleySurvivalNoFrameskip-v0',
    #     'SlimeVolleyNoFrameskip-v0', 'SlimeVolleyPixel-v0'
    # ],
    'pettingzoo': [
        'boxing_v2',
    ],
    # 'lasertag':
    # ['LaserTag-small2-v0', 'LaserTag-small3-v0', 'LaserTag-small4-v0'],
    # 'gym':
    # ['Pong-ram-v0', 'LunarLander-v2', 'CartPole-v1' ]  # 'HalfCheetah-v2' requires MuJoCo and mujoco-py to be installed
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
            test_env = make_env(env_args)
            try:
                test_env = make_env(env_args)
            except:
                fail_cnt += 1
                print(f'Failed to load env {env_name} in type {env_type}.')
            test_env.reset()
            test_env.render()
            print(f"{cnt-fail_cnt}/{cnt} succeed.")