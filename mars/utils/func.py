import yaml
from .data_struct import AttrDict
from .typing import Dict, Any
import collections.abc
import copy, os
from mars.utils.common import SelfplayBasedMethods, MetaStrategyMethods
from mars.rl.agents  import *
from multiprocessing.managers import BaseManager, NamespaceProxy
from mars.rl.common.storage import ReplayBuffer, ReservoirBuffer


def LoadYAML2Dict(yaml_file: str,
                  toAttr: bool = False,
                  mergeWith: str = 'mars/confs/default.yaml',
                  confs: Dict[str, Any] = {}) -> AttrDict:
    """ A function loading the hyper-parameters in yaml file into a dictionary.

    :param yaml_file: the yaml file name
    :type yaml_file: str
    :param toAttr: if True, transform the configuration dictionary into a class,
        such that each hyperparameter can be called with class.attribute instead of dict['attribute']; defaults to False
    :type toAttr: bool, optional
    :param mergeWith: if not None, merge the loaded yaml (with overwritting priority) with the yaml given by this path;
    for example, merging with default yaml file will fill those missing entries with those in defaulf configurations.
    :type mergeDefault: string or None, optional
    :param confs: input a dictionary of configurations from outside the function, defaults to {}
    :type confs: dict, optional
    :return: a dictionary of configurations, including all hyper-parameters for environment, algorithm and training/testing.
    :rtype: dict
    """
    if mergeWith is not None:
        with open(mergeWith) as f:
            default = yaml.safe_load(f)
        confs = UpdateDictAwithB(confs, default, withOverwrite=False)

    with open(yaml_file + '.yaml') as f:
        # use safe_load instead load
        loaded = yaml.safe_load(f)
    confs = UpdateDictAwithB(confs, loaded, withOverwrite=True)

    if toAttr:
        concat_dict = {
        }  # concatenate all types of arguments into one dictionary
        for k, v in confs.items():
            concat_dict.update(v)
        return AttrDict(concat_dict)
    else:
        return confs
        
def UpdateDictAwithB(
    A: Dict[str, Any],
    B: Dict[str, Any],
    withOverwrite: bool = True,
) -> None:
    """ Update the entries in dictionary A with dictionary B.

    :param A: a dictionary
    :type A: dict
    :param B: a dictionary
    :type B: dict
    :param withOverwrite: whether replace the same entries in A with B, defaults to False
    :type withOverwrite: bool, optional
    :return: none
    """
    # ensure original A, B is not changed
    A_ = copy.deepcopy(A)
    B_ = copy.deepcopy(B)
    if withOverwrite:
        InDepthUpdateDictAwithB(A_, B_)
    else:
        temp = copy.deepcopy(A_)
        InDepthUpdateDictAwithB(A_, B_)
        InDepthUpdateDictAwithB(A_, temp)

    return A_


def InDepthUpdateDictAwithB(
    A: Dict[str, Any],
    B: Dict[str, Any],
) -> None:
    """A function for update nested dictionaries A with B.

    :param A: a nested dictionary, e.g., dict, dict of dict, dict of dict of dict ...
    :type A: dict
    :param B: a nested dictionary, e.g., dict, dict of dict, dict of dict of dict ...
    :type B: dict
    :return: none
    """
    for k, v in B.items():
        if isinstance(v, collections.abc.Mapping):
            A[k] = InDepthUpdateDictAwithB(A.get(k, {}), v)
        else:
            A[k] = v
    return A

def get_general_args(env, method):
    [env_type, env_name] = env.split('_', 1) # only split at the first '_'
    path = f'mars/confs/{env_type}/{env_name}/'
    yaml_file = f'{env_type}_{env_name}_{method}'
    args = LoadYAML2Dict(path+yaml_file, toAttr=True, mergeWith='mars/confs/default.yaml')
    return args

def get_general_args_from_parser(parser_args):
    env = parser_args['env']
    method = parser_args['marl_method']
    [env_type, env_name] = env.split('_', 1) # only split at the first '_'
    path = f'mars/confs/{env_type}/{env_name}/'
    yaml_file = f'{env_type}_{env_name}_{method}'
    args = LoadYAML2Dict(path+yaml_file, toAttr=True, mergeWith='mars/confs/default.yaml')
    return args

def parse_args():
    """
    TODO: Parse user input arguments.
    """
    custom_parameters = [
        {"name": "--test", "action": "store_true", "default": False,
            "help": "Run trained policy, no training"},
        {"name": "--play", "action": "store_true", "default": False,
            "help": "Run trained policy, the same as test, can be used only by rl_games RL library"},
        {"name": "--resume", "type": int, "default": 0,
            "help": "Resume training or start testing from a checkpoint"},
        {"name": "--checkpoint", "type": str, "default": "Base",
            "help": "Path to the saved weights, only for rl_games RL library"},
        {"name": "--headless", "action": "store_true", "default": False,
            "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False,
            "help": "Use horovod for multi-gpu training, have effect only with rl_games RL library"},
        {"name": "--task", "type": str, "default": "Humanoid",
            "help": "Can be BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, Ingenuity"},
        {"name": "--task_type", "type": str,
            "default": "Python", "help": "Choose Python or C++"},
        {"name": "--rl_device", "type": str, "default": "cuda:0",
            "help": "Choose CPU or GPU device for inferencing policy network"},
        {"name": "--logdir", "type": str, "default": "logs/"},
        {"name": "--experiment", "type": str, "default": "Base",
            "help": "Experiment name. If used with --metadata flag an additional information about physics engine, sim device, pipeline and domain randomization will be added to the name"},
        {"name": "--metadata", "action": "store_true", "default": False,
            "help": "Requires --experiment flag, adds physics engine, sim device, pipeline info and if domain randomization is used to the experiment name provided by user"},
        {"name": "--cfg_train", "type": str,
            "default": "Base"},
        {"name": "--cfg_env", "type": str, "default": "Base"},
        {"name": "--num_envs", "type": int, "default": 0,
            "help": "Number of environments to create - override config file"},
        {"name": "--episode_length", "type": int, "default": 0,
            "help": "Episode length, by default is read from yaml config"},
        {"name": "--seed", "type": int, "help": "Random seed"},
        {"name": "--max_iterations", "type": int, "default": 0,
            "help": "Set a maximum number of training iterations"},
        {"name": "--steps_num", "type": int, "default": -1,
            "help": "Set number of simulation steps per 1 PPO iteration. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--minibatch_size", "type": int, "default": -1,
            "help": "Set batch size for PPO optimization step. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--randomize", "action": "store_true", "default": False,
            "help": "Apply physics domain randomization"},
        {"name": "--torch_deterministic", "action": "store_true", "default": False,
            "help": "Apply additional PyTorch settings for more deterministic behaviour"},
        {"name": "--algo", "type": str, "default": "happo",
            "help": "Choose an algorithm"},
        {"name": "--model_dir", "type": str, "default": "",
            "help": "Choose a model dir"}]

    # parse arguments
    from .isaacgym import gymutil
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    return args

def get_latest_file_in_folder(folder, id=None):
    idx_list = []
    file_list = []
    # get the latest model
    for f in os.listdir(folder):
        if f'_{id}' in f:  # ensure the model for the specified id exists
            idx_list.append(int(f.split('_')[0]))
            file_list.append(f)
    idx_list.sort()
    last_idx = idx_list[-1]
    file_path = folder+str(last_idx)
    if id is not None:
        file_path += '_'+str(id) 
    return file_path


def get_model_path(method, folder):
    if method in MetaStrategyMethods:
        file_path = folder
    elif method in SelfplayBasedMethods:
        file_path = get_latest_file_in_folder(folder)  # only one-side model is trained/saved
    else:
        file_path = get_latest_file_in_folder(folder, id=0)  # load from the first agent model of the two; most atari games does not share agent for left and right sides
    return file_path

def get_exploiter(exploiter_type: str, env, args):
    args.algorithm_spec['eps_decay'] = 100000  #100000   # fixed value for fair comparison; decay faster in exploitation than training

    if exploiter_type == 'DQN':
        ## This two lines are critical!
        args.algorithm_spec['episodic_update'] = False  # nash ppo has this as true, should be false since using DQN
        args.update_itr = 1  # nash-dqn has this 0.1, has to make it 1 for fair comparison with other methods
        if 'PPO' in args.algorithm:  # in PPO conf there is not network specification for DQN
            args.net_architecture = args.net_architecture['feature']  # make exploiter same net as the feature net in PPO
        exploiter = DQN(env, args)
        exploiter.reinit()
        exploitation_args = args

    elif exploiter_type == 'PPO':
        ppo_args = LoadYAML2Dict(f'mars/confs/{args.env_type}/ppo_exploit', toAttr=True, mergeWith=None)
        exploitation_args =  AttrDict(UpdateDictAwithB(args, ppo_args, withOverwrite=True))
        exploiter = PPO(env, exploitation_args)
        exploiter.reinit()

    return exploiter, exploitation_args


def multiprocess_buffer_register(args, method):
    """
    Register shared buffer for multiprocessing.
    """
    BaseManager.register('replay_buffer', ReplayBuffer)
    if method == 'nfsp':
        BaseManager.register('reservoir_buffer', ReservoirBuffer)
    manager = BaseManager()
    manager.start()
    add_components = {
        'replay_buffer': manager.replay_buffer(int(float(args.algorithm_spec['replay_buffer_size'])), \
            args.algorithm_spec['multi_step'], args.algorithm_spec['gamma'], args.num_envs, args.batch_size)  
    }
    
    if method == 'nfsp':
        add_components['reservoir_buffer'] = manager.reservoir_buffer(int(float(args.algorithm_spec['replay_buffer_size'])))  
    
    args.add_components = add_components

    return args

def multiprocess_conf(args, method):
    args.num_envs = 1  # this means one env per process; before buffer register
    args = multiprocess_buffer_register(args, method)
    args.max_episodes = int(args.max_episodes / args.num_process)
    args.multiprocess = True  # this is critical for launching multiprocess
    args.num_process = 5

    return args