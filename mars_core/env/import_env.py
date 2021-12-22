"""
Supported Environments in MARS:
Single-agent:
    * Openai Gym: https://gym.openai.com/
        type: 'gym'
        envs: see https://gym.openai.com/envs/
Multi-agent:
    * PettingZoo: https://github.com/PettingZoo-Team/PettingZoo
        type: 'pettingzoo'
        envs: [         
        'basketball_pong_v2', 'boxing_v1', 'combat_plane_v1', 'combat_tank_v1',
        'double_dunk_v2', 'entombed_competitive_v2', 'entombed_cooperative_v2',
        'flag_capture_v1', 'foozpong_v2', 'ice_hockey_v1', 'joust_v2',
        'mario_bros_v2', 'maze_craze_v2', 'othello_v2', 'pong_v2',
        'quadrapong_v3', 'space_invaders_v1', 'space_war_v1', 'surround_v1',
        'tennis_v2', 'video_checkers_v3', 'volleyball_pong_v2', 'warlords_v2',
        'wizard_of_wor_v2', 
        'dou_dizhu_v3', 'go_v3', 'leduc_holdem_v3',
        'rps_v1', 'texas_holdem_no_limit_v3', 'texas_holdem_v3',
        'tictactoe_v3', 'uno_v3']
    * LaserTag: https://github.com/younggyoseo/lasertag-v0
        type: 'lasertag'
        envs: ['LaserTag-small2-v0', 'LaserTag-small3-v0', 'LaserTag-small4-v0' ]
    * SlimeVolley: https://github.com/hardmaru/slimevolleygym 
        type: 'slimevolley'
        envs: ['SlimeVolley-v0', 'SlimeVolleySurvivalNoFrameskip-v0',
            'SlimeVolleyNoFrameskip-v0', 'SlimeVolleyPixel-v0']
"""
from typing import Dict
import pettingzoo
import slimevolleygym  # https://github.com/hardmaru/slimevolleygym
import gym
import supersuit
import numpy as np
from .wrappers.gym_wrappers import NoopResetEnv, MaxAndSkipEnv, WarpFrame, FrameStack, FireResetEnv, wrap_pytorch
from .wrappers.mars_wrappers import PettingzooClassicWrapper, PettingzooClassic_Iterate2Parallel,\
     Atari2AgentWrapper, SlimeVolleyWrapper, Dict2TupleWrapper
from .wrappers.vecenv_wrappers import DummyVectorEnv, SubprocVectorEnv
from .wrappers.lasertag_wrappers import LaserTagWrapper 


# PettingZoo envs
pettingzoo_envs = {
    'atari': [
        'basketball_pong_v2', 'boxing_v1', 'combat_plane_v1', 'combat_tank_v1',
        'double_dunk_v2', 'entombed_competitive_v2', 'entombed_cooperative_v2',
        'flag_capture_v1', 'foozpong_v2', 'ice_hockey_v1', 'joust_v2',
        'mario_bros_v2', 'maze_craze_v2', 'othello_v2', 'pong_v2',
        'quadrapong_v3', 'space_invaders_v1', 'space_war_v1', 'surround_v1',
        'tennis_v2', 'video_checkers_v3', 'volleyball_pong_v2', 'warlords_v2',
        'wizard_of_wor_v2'
    ],

    'classic': [
        'dou_dizhu_v4', 'go_v5', 'leduc_holdem_v4', 'rps_v2',
        'texas_holdem_no_limit_v4', 'texas_holdem_v4', 'tictactoe_v3', 'uno_v4'
    ]
}

for env_type, envs in pettingzoo_envs.items():
    for env_name in envs:
        try:
            exec("from pettingzoo.{} import {}".format(env_type.lower(), env_name))
            # print(f"Successfully import {env_type} env in PettingZoo: ", env_name)
        except:
            print("Cannot import pettingzoo env: ", env_name)

def _create_single_env(env_name: str, env_type: str, args: Dict):
    """A function create a single environment object given the name and type of 
    environment, as well as necessary arguments.

    :param env_name: the name of environment
    :type env_name: str
    :param env_type: the type of environment
    :type env_type: str
    :param args: necessary arguments for specifying the environment
    :type args: dict
    :return: the instantiation of an environment
    :rtype: object
    """
    if args.num_envs > 1:
        keep_info = True  # keep_info True to maintain dict type for parallel envs (otherwise cannot pass VectorEnv wrapper)
    else:
        keep_info = False

    if env_type == 'slimevolley':
        env = gym.make(env_name)
        if env_name in ['SlimeVolleySurvivalNoFrameskip-v0', 'SlimeVolleyNoFrameskip-v0', 'SlimeVolleyPixel-v0']:
            # For image-based envs, apply following wrappers (from gym atari) to achieve pettingzoo style env, 
            # or use supersuit (requires input env to be either pettingzoo or gym env).
            # same as: https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ppo_pixel.py
            # TODO Note: this cannot handle the two obervations in above SlimeVolley envs, 
            # since the wrappers are for single agent.
            if env_name != 'SlimeVolleyPixel-v0':
                env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = WarpFrame(env) 
            # #env = ClipRewardEnv(env)
            env = FrameStack(env, 4)

        env = SlimeVolleyWrapper(env, args.against_baseline)  # slimevolley to pettingzoo style
        env = Dict2TupleWrapper(env, keep_info=keep_info)  # pettingzoo to nfsp style, keep_info True to maintain dict type for parallel envs

    elif env_type == 'pettingzoo':
        if env_name in pettingzoo_envs['atari']:
            if args.ram:
                obs_type = 'ram'
            else:
                obs_type = 'rgb_image'

            # initialize the env
            env = eval(env_name).parallel_env(obs_type=obs_type, full_action_space=False)
            # env = supersuit.agent_indicator_v0(env) # TODO for selfplay, agent from two sides can use the same model (but it needs to see samples from two sides); see https://github.com/PettingZoo-Team/PettingZoo/issues/423
            env_agents = env.unwrapped.agents  # this cannot go through supersuit wrapper, so get it first and reassign it

            # assign necessary wrappers
            if obs_type == 'rgb_image':
                env = supersuit.max_observation_v0(env, 2)  # as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames to deal with frame flickering
                env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25) # repeat_action_probability is set to 0.25 to introduce non-determinism to the system
                env = supersuit.frame_skip_v0(env, 4) # skip frames for faster processing and less control to be compatable with gym, use frame_skip(env, (2,5))
                env = supersuit.resize_v0(env, 84, 84) # downscale observation for faster processing
                env = supersuit.frame_stack_v1(env, 4) # allow agent to see everything on the screen despite Atari's flickering screen problem
            else:
                env = supersuit.frame_skip_v0(env, 4)  # RAM version also need frame skip, essential for boxing-v1, etc
                    
            # normalize the observation of Atari for both image or RAM 
            env = supersuit.dtype_v0(env, 'float32') # need to transform uint8 to float first for normalizing observation: https://github.com/PettingZoo-Team/SuperSuit
            env = supersuit.normalize_obs_v0(env, env_min=0, env_max=1) # normalize the observation to (0,1)

            # assign observation and action spaces
            env.observation_space = list(env.observation_spaces.values())[0]
            env.action_space = list(env.action_spaces.values())[0]
            env.agents = env_agents
            env = Dict2TupleWrapper(env, keep_info=keep_info) 

        elif env_name in pettingzoo_envs['classic']:
            if env_name in ['rps_v2', 'rpsls_v1']:
                env = eval(env_name).parallel_env()
                env = PettingzooClassicWrapper(env, observation_mask=1.)
            else: # only rps_v1 can use parallel_env at present
                env = eval(env_name).env()
                env = PettingzooClassic_Iterate2Parallel(env, observation_mask=None)  # since Classic games do not support Parallel API yet
                
            env = Dict2TupleWrapper(env, keep_info=keep_info)

    elif env_type == 'lasertag':
        import lasertag  # this is essential
        env = gym.make(env_name)
        env = wrap_pytorch(env) 
        env = LaserTagWrapper(env)

    elif env_type == 'gym':
        try:
            env = gym.make(env_name)
        except:
            print(f"Error: No such env in Openai Gym: {env_name}!") 
        # may need more wrappers here, e.g. Pong-ram-v0 need scaled observation!
        # Ref: https://towardsdatascience.com/deep-q-network-dqn-i-bce08bdf2af
        env = Atari2AgentWrapper(env)

    else:
        print(f"Error: {env_name} environment in type {env_type} not found!")
        return 

    print(f'Load {env_name} environment in type {env_type}.')  
    print(f'Env observation space: {env.observation_space} action space: {env.action_space}')  
    return env

def make_env(args):
    """A function for creating all environments, could be multiple if using parallel settings.

    :param args: necessary arguments for specifying the environment
    :type args: dict
    :return: env or envs
    :rtype: object or VectorEnv
    """
    env_name = args.env_name
    env_type = args.env_type
    print(env_name, env_type)

    if args.num_envs == 1:
        env = _create_single_env(env_name, env_type, args)  
    else:
        VectorEnv = [DummyVectorEnv, SubprocVectorEnv][1]  
        env = VectorEnv([lambda: _create_single_env(env_name, env_type, args) for _ in range(args.num_envs)])
    if isinstance(args.seed, (int, list)):
        env.seed(args.seed)  # seed can be either int or list of int
    elif args.seed == 'random':
        if args.num_envs > 1:
            random_seed = [int(seed) for seed in np.random.randint(1,100, args.num_envs)]
        else:
            random_seed = int(np.random.randint(1,100))
        print(f"random seed: {random_seed}")
        env.seed(random_seed)
    return env
