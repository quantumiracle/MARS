import argparse
import sys
from mars.utils.func import LoadYAML2Dict
from mars.utils.wandb import init_wandb

def get_parser_args():
    ''' deprecated '''
    parser = argparse.ArgumentParser(description='Arguments of the general launching script for MARS.')
    # env args
    parser.add_argument('--env', type=str, default=None, help='environment type and name')
    parser.add_argument('--num_envs', type=int, default=1, help='number of environments for parallel sampling')
    parser.add_argument('--ram', type=bool, default=False, help='use RAM observation')
    parser.add_argument('--render', type=bool, default=False, help='render the scene')
    parser.add_argument('--seed', type=str, default='random', help='random seed')
    parser.add_argument('--record_video', type=bool, default=False, help='whether recording the video')

    # agent args
    parser.add_argument('--algorithm', type=str, default=None, help='algorithm name')
    parser.add_argument('--algorithm_spec.dueling', type=bool, default=False, help='DQN: dueling trick')
    parser.add_argument('--algorithm_spec.replay_buffer_size', type=int, default=1e5, help='DQN: replay buffer size')
    parser.add_argument('--algorithm_spec.gamma', type=float, default=0.99, help='DQN: discount factor')
    parser.add_argument('--algorithm_spec.multi_step', type=int, default=1, help='DQN: multi-step return')
    parser.add_argument('--algorithm_spec.target_update_interval', type=bool, default=False, help='DQN: steps skipped for target network update')
    parser.add_argument('--algorithm_spec.eps_start', type=float, default=1, help='DQN: epsilon-greedy starting value')
    parser.add_argument('--algorithm_spec.eps_final', type=float, default=0.001, help='DQN: epsilon-greedy ending value')
    parser.add_argument('--algorithm_spec.eps_decay', type=float, default=5000000, help='DQN: epsilon-greedy decay interval')

    # train args
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for update')
    parser.add_argument('--max_episodes', type=int, default=50000, help='maximum episodes for rollout')
    parser.add_argument('--max_steps_per_episode', type=int, default=300, help='maximum steps per episode')
    parser.add_argument('--train_start_frame', type=int, default=0, help='start frame for training (not update when warmup)')
    parser.add_argument('--method', dest='marl_method', type=str, default=None, help='method name')
    parser.add_argument('--save_id', type=str, default='0', help='identification number for each run')

    # get all argument values
    parser_args = parser.parse_args()

    # get all default argument values
    defaults = vars(parser.parse_args([]))

    # get non-default argument values TODO
    print(defaults)
    print('parser args: ', parser_args)

    return parser_args

def get_default_args(env, method):
    print(env)
    [env_type, env_name] = env.split('_', 1) # only split at the first '_'
    path = f'mars/confs/{env_type}/{env_name}/'
    yaml_file = f'{env_type}_{env_name}_{method}'
    args = LoadYAML2Dict(path+yaml_file, toAttr=True, mergeWith='mars/confs/default.yaml')
    return args

def get_default_args_given_args(parser_args):
    env = parser_args['env']
    method = parser_args['marl_method']
    [env_type, env_name] = env.split('_', 1) # only split at the first '_'
    path = f'mars/confs/{env_type}/{env_name}/'
    yaml_file = f'{env_type}_{env_name}_{method}'
    args = LoadYAML2Dict(path+yaml_file, toAttr=True, mergeWith='mars/confs/default.yaml')
    return args


def get_args():
    config_doc = """
    #######################################################################################
    Descriptions of configurations:
    ---------------------------------------------------------------------------------------
    Environment args:
    * '--env',                   type=str,  default=None,     description='environment type and name'
    * '--num_envs',              type=int,  default=1,        description='number of environments for parallel sampling'
    * '--ram',                   type=bool, default=False,    description='use RAM observation'
    * '--render',                type=bool, default=False,    description='render the scene'
    * '--seed',                  type=str,  default='random', description='random seed'
    * '--record_video',          type=bool, default=False,    description='whether recording the video'
    * '--record_video_interval', type=int,  default=False,    description='how often (skipped steps) to record the video'
    * '--record_video_length',   type=int,  default=False,    description='how long (steps) to record the video'

    Agent args:
    * '--algorithm',                             type=str,   default=None,    description='algorithm name'
    * '--algorithm_spec.dueling',                type=bool,  default=False,   description='DQN: dueling trick'
    * '--algorithm_spec.replay_buffer_size',     type=int,   default=1e5,     description='DQN: replay buffer size'
    * '--algorithm_spec.gamma',                  type=float, default=0.99,    description='DQN: discount factor'
    * '--algorithm_spec.multi_step',             type=int,   default=1,       description='DQN: multi-step return'
    * '--algorithm_spec.target_update_interval', type=bool,  default=False,   description='DQN: steps skipped for target network update'
    * '--algorithm_spec.eps_start',              type=float, default=1,       description='DQN: epsilon-greedy starting value'
    * '--algorithm_spec.eps_final',              type=float, default=0.001,   description='DQN: epsilon-greedy ending value'
    * '--algorithm_spec.eps_decay',              type=float, default=5000000, description='DQN: epsilon-greedy decay interval'

    Training args:
    * '--num_process',                   type=int,   default=1,      description='use multiprocessing for both sampling and update'
    * '--batch_size',                    type=int,   default=128,    description='batch size for update'
    * '--max_episodes',                  type=int,   default=50000,  description='maximum episodes for rollout'
    * '--max_steps_per_episode',         type=int,   default=300,    description='maximum steps per episode'
    * '--train_start_frame',             type=int,   default=0,      description='start frame for training (not update when warmup)'
    * '--method', dest='marl_method',    type=str,   default=None,   description='method name'
    * '--save_id',                       type=str,   default='0',    description='identification number for each run'
    * '--optimizer',                     type=str,   default='adam', description='choose optimizer'
    * '--batch_size',                    type=int,   default=128,    description='batch size for update'
    * '--learning_rate',                 type=float, default=1e-4,   description='learning rate'
    * '--device',                        type=str,   default='gpu',  description='computation device for model optimization'
    * '--update_itr',                    type=int,   default=1,      description='number of updates per step'
    * '--log_avg_window',                type=int,   default=20,     description='window length for averaging the logged results'
    * '--log_interval',                  type=int,   default=20,     description='interval for logging'
    * '--test',                          type=bool,  default=False,  description='whether in test mode'
    * '--exploit',                       type=bool,  default=False,  description='whether in exploitation mode'
    * '--load_id',                       type=str,   default='0',    description='index of the model to load'
    * '--load_model_full_path',          type=str,   default='/',    description='full path of the model to load'
    * '--multiprocess',                  type=bool,  default=False,  description='whether using multiprocess or not'
    * '--eval_models',                   type=bool,  default=False,  description='evalutation models during training (only for specific methods)'
    * '--save_path',                     type=str,   default='/',    description='path to save models and logs'
    * '--save_interval',                 type=int,   default=2000,   description='episode interval to save models'
    * '--wandb_activate',                type=bool,  default=False,  description='activate wandb for logging'
    * '--wandb_entity',                  type=str,   default='',     description='wandb entity'
    * '--wandb_project',                 type=str,   default='',     description='wandb project'
    * '--wandb_group',                   type=str,   default='',     description='wandb group'
    * '--wandb_name',                    type=str,   default='',     description='wandb name'
    * '--net_architecture.hidden_dim_list',   type=str, default='[128, 128, 128]', description='list of hidden dimensions for model'
    * '--net_architecture.hidden_activation', type=str, default='ReLU',            description='hidden activation function'
    * '--net_architecture.output_activation', type=str, default=False,             description='output activation function'
    * '--net_architecture.hidden_activation', type=str, default='ReLU',            description='hidden activation function'
    * '--net_architecture.channel_list',      type=str, default='[8, 8, 16]',      description='list of channels for CNN'
    * '--net_architecture.kernel_size_list',  type=str, default='[4, 4, 4]',       description='list of kernel sizes for CNN'
    * '--net_architecture.stride_list',       type=str, default='[2, 1, 1]',       description='list of strides for CNN'
    * '--marl_spec.min_update_interval',      type=int, default=20,                description='mininal opponent update interval in unit of episodes'
    * '--marl_spec.score_avg_window',         type=int, default=10,                description='the length of window for averaging the score values'
    * '--marl_spec.global_state',             type=bool, default=False,            description='whether using global observation'

    #########################################################################################
    """
    mapping_path = []
    parsed_ids = []

    # get env and methods
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--env':
            arg_env = sys.argv[1:][i+1]
            parsed_ids.extend([i, i+1])
        if arg == '--method':
            arg_method = sys.argv[1:][i+1]
            parsed_ids.extend([i, i+1])
        if arg == '-W': # a special one, needs to be handled first
            parsed_ids.extend([i, i+1])

    first_arg = sys.argv[1:][0]
    if first_arg == '--help' or first_arg == '-h':
        print(config_doc)
        exit()
    else:
        # get default args
        default_args = get_default_args(arg_env, arg_method)
        # print('default: ', default_args)
        
        # overwrite default with user input args
        for i, arg in enumerate(sys.argv[1:]):
            if i not in parsed_ids:
                if arg.startswith('--'):
                    mapping_path = arg[2:].split('.')
                else:        
                    ind = default_args
                    for p in mapping_path[:-1]:
                        ind = ind[p]
                    try:
                        ind[mapping_path[-1]] = eval(arg)
                    except:
                        ind[mapping_path[-1]] = arg

    print(default_args)  # args after overwriting

    # initialize wandb if necessary
    if default_args.wandb_activate:
        if len(default_args.wandb_project) == 0:
            default_args.wandb_project = 'Pettingzoo_MARS'
        if len(default_args.wandb_group) == 0:
            default_args.wandb_group = str(default_args.save_id)
        if len(default_args.wandb_name) == 0:
            try: # multi-agent
                default_args.wandb_name = '_'.join((default_args.env_type, default_args.env_name, default_args.marl_method, str(default_args.save_id)))
            except: # single-agent
                default_args.wandb_name = '_'.join((default_args.env_type, default_args.env_name, default_args.algorithm, str(default_args.save_id)))
        init_wandb(default_args)
        
    return default_args
