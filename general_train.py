from mars.utils.func import LoadYAML2Dict
from mars.env.import_env import make_env
from mars.rollout import rollout
from mars.rl.agents import *
from mars.rl.agents.multiagent import MultiAgent
from mars.utils.args_parser import get_args
from mars.utils.data_struct import AttrDict

def launch():
    args = get_args()
    print('args: ', args)

    ### Create env
    env = make_env(args)
    print(env)

    ### Specify models for each agent     
    model1 = eval(args.algorithm)(env, args)
    model2 = eval(args.algorithm)(env, args)

    model = MultiAgent(env, [model1, model2], args)

    ### Rollout
    rollout(env, model, args, args.save_id)

if __name__ == '__main__':
    launch()  # vars: Namespace -> dict