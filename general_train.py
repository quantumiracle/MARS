from mars.utils.func import LoadYAML2Dict
from mars.env.import_env import make_env
from mars.rollout import rollout
from mars.rl.agents import *
from mars.rl.agents.multiagent import MultiAgent
from mars.utils.func import get_general_args
from mars.utils.data_struct import AttrDict
from mars.utils.args_parser import get_parser_args

def launch(parser_args):
    args = get_general_args(parser_args['env'], parser_args['marl_method'])
    args = AttrDict({**args, **parser_args})  # overlap default with user input args
    print('args: ', args)
    # return 0

    ### Create env
    env = make_env(args)
    print(env)

    ### Specify models for each agent     
    model1 = eval(args.algorithm)(env, args)
    model2 = eval(args.algorithm)(env, args)

    model = MultiAgent(env, [model1, model2], args)

    ### Rollout
    rollout(env, model, args, parser_args['save_id'])

if __name__ == '__main__':
    parser_args = get_parser_args()
    launch(vars(parser_args))  # vars: Namespace -> dict