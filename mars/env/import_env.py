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
        'basketball_pong_v3', 'boxing_v2', 'combat_plane_v2', 'combat_tank_v2',
        'double_dunk_v3', 'entombed_competitive_v3', 'entombed_cooperative_v3',
        'flag_capture_v2', 'foozpong_v3', 'ice_hockey_v2', 'joust_v3',
        'mario_bros_v3', 'maze_craze_v3', 'othello_v3', 'pong_v3',
        'quadrapong_v4', 'space_invaders_v2', 'space_war_v2', 'surround_v2',
        'tennis_v3', 'video_checkers_v4', 'volleyball_pong_v3', 'warlords_v3',
        'wizard_of_wor_v3'
        'dou_dizhu_v4', 'go_v5', 'leduc_holdem_v4', 'rps_v2',
        'texas_holdem_no_limit_v6', 'texas_holdem_v4', 'tictactoe_v3', 'uno_v4']
    * LaserTag: https://github.com/younggyoseo/lasertag-v0
        type: 'lasertag'
        envs: ['LaserTag-small2-v0', 'LaserTag-small3-v0', 'LaserTag-small4-v0' ]
    * SlimeVolley: https://github.com/hardmaru/slimevolleygym 
        type: 'slimevolley'
        envs: ['SlimeVolley-v0', 'SlimeVolleySurvivalNoFrameskip-v0',
            'SlimeVolleyNoFrameskip-v0', 'SlimeVolleyPixel-v0']
"""
from typing import Dict
import gym
import slimevolleygym
import numpy as np


from .wrappers.gym_wrappers import NoopResetEnv, MaxAndSkipEnv, WarpFrame, FrameStack, NormalizeReward, TransformObservation, NormalizeObservation, FireResetEnv, wrap_pytorch
from .wrappers.mars_wrappers import PettingzooClassicWrapper, PettingzooClassic_Iterate2Parallel,\
     Gym2AgentWrapper, Gym2AgentAdversarialWrapper, SlimeVolleyWrapper, Dict2TupleWrapper, RoboSumoWrapper, SSVecWrapper, ZeroSumWrapper, zero_sum_reward_filer
from .wrappers.vecenv_wrappers import DummyVectorEnv, SubprocVectorEnv
from .wrappers.lasertag_wrappers import LaserTagWrapper
from .mdp import attack, combinatorial_lock, arbitrary_mdp, arbitrary_richobs_mdp

# gym verison change cannot be done on the fly (in runtime)
# def install_package(package):
#     import importlib
#     try:
#         importlib.import_module(package)
#     except ImportError:
#         import pip
#         pip.main(['install', package])
#     finally:
#         import gym
#         print(f'gym version: {gym.__version__}')
#         # if '==' in package:
#         #     name = package.split('=')[0]
#         # else:
#         #     name = package
#         # globals()[name] = importlib.import_module(name)


# PettingZoo envs
pettingzoo_envs = {
    'atari': [
        'basketball_pong_v3', 'boxing_v2', 'combat_plane_v2', 'combat_tank_v2',
        'double_dunk_v3', 'entombed_competitive_v3', 'entombed_cooperative_v3',
        'flag_capture_v2', 'foozpong_v3', 'ice_hockey_v2', 'joust_v3',
        'mario_bros_v3', 'maze_craze_v3', 'othello_v3', 'pong_v3',
        'quadrapong_v4', 'space_invaders_v2', 'space_war_v2', 'surround_v2',
        'tennis_v3', 'video_checkers_v4', 'volleyball_pong_v3', 'warlords_v3',
        'wizard_of_wor_v3'
    ],

    'classic': [
        'dou_dizhu_v4', 'go_v5', 'leduc_holdem_v4', 'rps_v2',
        'texas_holdem_no_limit_v6', 'texas_holdem_v4', 'tictactoe_v3', 'uno_v4'
    ]
}

for env_type, envs in pettingzoo_envs.items():
    for env_name in envs:
        try:
            exec("from pettingzoo.{} import {}".format(env_type.lower(), env_name))
            # print(f"Successfully import {env_type} env in PettingZoo: ", env_name)
        except:
            print("Cannot import pettingzoo env: ", env_name)

def _create_single_env(env_name: str, env_type: str, ss_vec: True, args: Dict):
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
    if env_type not in ['robosumo', 'slimevolley']: # robosumo uses a different version of gym (0.16), slimevolley requires gym<=0.19 (0.18), conflicting with supersuit (gym==0.22)
        from .wrappers.pettingzoo_parallel_reward_lambda import reward_lambda_v1

    if args.num_envs > 1:
        keep_info = True  # keep_info True to maintain dict type for parallel envs (otherwise cannot pass VectorEnv wrapper)
    else:
        keep_info = False

    if env_type == 'slimevolley':
        if not args.ram:
            env_name = 'SlimeVolleyPixel-v0'
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
        # env = reward_lambda_v1(env, zero_sum_reward_filer)

    elif env_type == 'pettingzoo':
        import supersuit
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
                # env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25) # repeat_action_probability is set to 0.25 to introduce non-determinism to the system
                env = supersuit.color_reduction_v0(env, mode="B")
                env = supersuit.frame_skip_v0(env, 4) # skip frames for faster processing and less control to be compatable with gym, use frame_skip(env, (2,5))
                env = supersuit.resize_v1(env, 84, 84) # downscale observation for faster processing
                env = supersuit.frame_stack_v1(env, 4) # allow agent to see everything on the screen despite Atari's flickering screen problem
            else:
                env = supersuit.frame_skip_v0(env, 4)  # RAM version also need frame skip, essential for boxing-v1, etc
                    
            # normalize the observation of Atari for both image or RAM 
            env = supersuit.dtype_v0(env, 'float32') # need to transform uint8 to float first for normalizing observation: https://github.com/PettingZoo-Team/SuperSuit
            env = supersuit.normalize_obs_v0(env, env_min=0, env_max=1) # normalize the observation to (0,1)

            env = reward_lambda_v1(env, zero_sum_reward_filer)

            # assign observation and action spaces
            if not ss_vec:
                env.observation_space = list(env.observation_spaces.values())[0]
                env.action_space = list(env.action_spaces.values())[0]
                env.agents = env_agents
                env = Dict2TupleWrapper(env, keep_info=keep_info) 

            env.agents = env_agents
            
        elif env_name in pettingzoo_envs['classic']:
            if env_name in ['rps_v2', 'rpsls_v1']:
                env = eval(env_name).parallel_env()
                env = PettingzooClassicWrapper(env, observation_mask=1.)
            else: # only rps_v1 can use parallel_env at present
                env = eval(env_name).env()
                env = PettingzooClassic_Iterate2Parallel(env, observation_mask=None)  # since Classic games do not support Parallel API yet
               
            env = Dict2TupleWrapper(env, keep_info=keep_info)
            # env = reward_lambda_v1(env, zero_sum_reward_filer)

    elif env_type == 'lasertag':
        import lasertag  # this is essential
        env = gym.make(env_name)
        env = wrap_pytorch(env) 
        env = LaserTagWrapper(env)

    elif env_type == 'robosumo':
        # robosumo requires gym==0.16;
        # with updated version (https://github.com/Robot-Learning-Library/robosumo_gym23): gym==0.23 can be used
        import robosumo.envs
        env = gym.make(env_name)

        if args.record_video:
            mode = 'rgb_array' # this willl return image from render() thus recording, but not render scene
        elif args.render:
            mode = 'human'  # requies: export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so; this will render scene but not return image
        else:
            mode = 'rgb_array'
        env = RoboSumoWrapper(env, mode)
        # these wrappers work for multi-agent as well, stats are maintained for each agent individually
        env = NormalizeObservation(env) # according to original paper: https://arxiv.org/pdf/1710.03641.pdf
        env = TransformObservation(env, lambda obs: np.clip(obs, -5., 5.))
        env = NormalizeReward(env)
        # env = ZeroSumWrapper(env)

    elif env_type == 'gym':
        try:
            env = gym.make(env_name)
        except:
            print(f"Error: No such env in Openai Gym: {env_name}!") 
        # may need more wrappers here, e.g. Pong-ram-v0 need scaled observation!
        # Ref: https://towardsdatascience.com/deep-q-network-dqn-i-bce08bdf2af
        env = gym.wrappers.RecordEpisodeStatistics(env)  # bypass the reward normalization to record episodic return
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env) 
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)  # this can be critical for algo to work
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        
        if args.adversarial:
            env = Gym2AgentAdversarialWrapper(env)
        else:
            env = Gym2AgentWrapper(env)
     

    elif env_type == 'safetygym':
        import safety_gym
        try:
            env = gym.make(env_name)
        except:
            print(f"Error: No such env in Safety Gym: {env_name}!")       
        env = Gym2AgentWrapper(env) 

    elif env_type == 'mdp':
        if env_name == 'arbitrary_mdp':
            env = arbitrary_mdp
            # env.NEsolver()
        elif env_name == 'arbitrary_richobs_mdp':
            env = arbitrary_richobs_mdp
        elif env_name == 'attack':
            env = attack
        elif env_name == 'combinatorial_lock':
            env = combinatorial_lock
        else:
            raise NotImplementedError
    else:
        print(f"Error: {env_name} environment in type {env_type} not found!")
        raise NotImplementedError

    print(f'Load {env_name} environment in type {env_type}.')  
    print(f'Env observation space: {env.observation_space} action space: {env.action_space}')  

    return env

def make_env(args, ss_vec=True):
    """A function for creating all environments, could be multiple if using parallel settings.

    :param args: necessary arguments for specifying the environment
    :type args: dict
    :return: env or envs
    :rtype: object or VectorEnv
    """
    env_name = args.env_name
    env_type = args.env_type
    print(env_name, env_type)

    # different gym dependency, this will not work on the fly
    # if env_type == 'pettingzoo':
    #     install_package('gym==0.23')
    # elif env_type == 'slimevolley':
    #     install_package('gym==0.18')
    # elif env_type == 'robosumo':
    #     install_package('gym==0.16')

    # video recorder: https://github.com/openai/gym/blob/master/gym/wrappers/record_video.py
    if not 'record_video_interval' in args.keys():
        record_video_interval = int(1e3) 
    else:
        record_video_interval =  int(args.record_video_interval)
    if not 'record_video_length' in args.keys():
        record_video_length = 300 # by default 0 it records entire episode, otherwise >0 specify the steps
    else:
        record_video_length = int(args.record_video_length)
    print(f'record video: interval {record_video_interval}, length {record_video_length}')

    if args.num_process > 1 or args.num_envs == 1: # if multiprocess, each process can only work with one env separately
        env = _create_single_env(env_name, env_type, False, args)  
        # gym has to be 0.23.1 to successfully record video here!
        if args.record_video: # Ref: https://github.com/openai/gym/pull/2300
            env = gym.wrappers.RecordVideo(env, f"data/videos/{args.env_type}_{args.env_name}_{args.algorithm}_{args.save_id}",\
                    # step_trigger=lambda step: step % record_video_interval == 0, # record the videos every 10000 steps
                    episode_trigger=lambda episode: episode % record_video_interval == 0, # record the videos every * episodes
                    video_length=record_video_length,  # record full episode if commented
                    )
    else:
        if env_type == 'pettingzoo' and ss_vec:
            import supersuit
            single_env = _create_single_env(env_name, env_type, True, args)
            vec_env = supersuit.pettingzoo_env_to_vec_env_v1(single_env)
            import multiprocessing
            # env = supersuit.concat_vec_envs_v1(vec_env, args.num_envs, num_cpus=multiprocessing.cpu_count(), base_class="gym")  # true number of envs will be args.num_envs
            env = supersuit.concat_vec_envs_v1(vec_env, args.num_envs, num_cpus=0, base_class="gym")  # true number of envs will be args.num_envs
            # env = gym.wrappers.RecordEpisodeStatistics(env)

            if args.record_video:
                env.is_vector_env = True
                env = gym.wrappers.RecordVideo(env, f"data/videos/{args.env_type}_{args.env_name}_{args.algorithm}_{args.save_id}",\
                        # step_trigger=lambda step: step % record_video_interval == 0, # record the videos every 10000 steps
                        episode_trigger=lambda episode: episode % record_video_interval == 0, # record the videos every * episodes
                        video_length=record_video_length
                        )  
            # print(args.num_envs, env.num_envs)
            env.num_agents = single_env.num_agents
            env.agents = single_env.agents
            env = SSVecWrapper(env)

        else:
            VectorEnv = [DummyVectorEnv, SubprocVectorEnv][1]  
            single_env = _create_single_env(env_name, env_type, False, args)
            env = VectorEnv([lambda: single_env for _ in range(args.num_envs)])
            if args.record_video:
                env.is_vector_env = True
                # record single env if multiple envs are used TODO
                env.metadata = single_env.metadata
                single_env = gym.wrappers.RecordVideo(single_env, f"data/videos/{args.env_type}_{args.env_name}_{args.algorithm}_{args.save_id}",\
                        # step_trigger=lambda step: step % record_video_interval == 0, # record the videos every 10000 steps
                        episode_trigger=lambda episode: episode % record_video_interval == 0, # record the videos every * episodes
                        video_length=record_video_length
                        ) 
            # avoid duplicating
            env.num_agents = single_env.num_agents
            env.agents = single_env.agents
    
        # print('metadata: ', env.metadata)
        # print(env[0].metadata)

    if isinstance(args.seed, (int, list)):
        env.seed(args.seed)  # seed can be either int or list of int
    elif args.seed == 'random':
        np.random.seed(seed=None)  # make np random
        if args.num_process > 1 or args.num_envs == 1:
            random_seed = int(np.random.randint(1,1000))
        else:  # more than one env
            random_seed = [int(seed) for seed in np.random.randint(1,1000, args.num_envs)]
        print(f"random seed: {random_seed}")
        env.seed(random_seed)

    return env
