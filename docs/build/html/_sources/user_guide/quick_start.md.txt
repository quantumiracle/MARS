# A Quick Start

## Single-agent RL

* Train a model on single-agent game, e.g. *CartPole-v1 OpenAI Gym*, using PPO algorithm:

  ```python
  from mars.utils.func import LoadYAML2Dict
  from mars.env.import_env import make_env
  from mars.rollout import rollout
  from mars.rl.agents import *
  from mars.rl.agents.multiagent import MultiAgent
  
  ### Load configurations
  yaml_file = 'mars/confs/gym_cartpolev1_dqn' #PATH TO YAML
  args = LoadYAML2Dict(yaml_file, toAttr=True)
  
  ### Create env
  env = make_env(args)
  print(env)
  
  ### Specify models for each agent
  model = eval(args.algorithm)(env, args)
  
  model = MultiAgent(env, [model], args)
  
  ### Rollout
  rollout(env, model, args)
  ```

* Test the trained model:

  ```python
  from mars.utils.func import LoadYAML2Dict
  from mars.env.import_env import make_env
  from mars.rollout import rollout
  from mars.rl.agents import *
  from mars.rl.agents.multiagent import MultiAgent
  
  ### Load configurations
  yaml_file = 'mars/confs/gym_cartpolev1_dqn' #PATH TO YAML
  args = LoadYAML2Dict(yaml_file, toAttr=True)
  
  ## Change/specify some arguments if necessary
  args.test = True  # the test mode will automatically fix all models
  args.render = True
  args.load_model_full_path = 'PATH TO THE TRAINED MODEL' # or use args.load_model_idx
  
  ### Create env
  env = make_env(args)
  
  ### Specify models for each agent
  model = eval(args.algorithm)(env, args)
  
  model = MultiAgent(env, [model], args)
  
  ### Rollout
  rollout(env, model, args)
  ```
  
  

## Multi-agent RL

* (Single-process) Train a model on two-agent game, e.g. *boxing-v1 PettingZoo*, using self-play with DQN algorithm:

  ```python
  from mars.utils.func import LoadYAML2Dict
  from mars.env.import_env import make_env
  from mars.rollout import rollout
  from mars.rl.agents import *
  from mars.rl.agents.multiagent import MultiAgent
  
  ### Load configurations
  yaml_file = 'mars_core/confs/pettingzoo/boxing_v1/pettingzoo_boxing_v1_selfplay'
  args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)
  args.multiprocess = False
  
  ### Create env
  env = make_env(args)
  
  ### Specify models for each agent
  model1 = eval(args.algorithm)(env, args)
  model2 = eval(args.algorithm)(env, args)
  # model1.fix()  # fix a model if you don't want it to learn
  
  model = MultiAgent(env, [model1, model2], args)
  
  ### Rollout
  rollout(env, model, args)
  ```
  
* (Multi-process-separate for sampling and model update) Train a model on two-agent game, e.g. *boxing-v1 PettingZoo*, using self-play with DQN algorithm:

  ```python
  import argparse
  import copy
  import torch
  torch.multiprocessing.set_start_method('forkserver', force=True)
  from multiprocessing import Process, Queue
  from mars.env.import_env import make_env
  from mars.rl.agents import *
  from mars.rl.agents.multiagent import MultiAgent
  from mars.utils.func import get_general_args, multiprocess_conf
  from rolloutExperience import rolloutExperience
  from updateModel import updateModel
  
  ### Load configurations
  yaml_file = 'mars_core/confs/pettingzoo/boxing_v1/pettingzoo_boxing_v1_selfplay'
  args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)
  num_envs = args.num_envs  # this will be changed to 1 later
  multiprocess_conf(args, method)
  
  ### Create env
  env = make_env(args)
  print(env)
  
  ### Specify models for each agent
  args = multiprocess_conf(args, method)
  model1 = eval(args.algorithm)(env, args)
  model2 = eval(args.algorithm)(env, args)
  
  model = MultiAgent(env, [model1, model2], args)
  env.close()  # this env is only used for creating other intantiations
  
  processes = []
  print(args)
  
  ### launch multiple sample rollout processes
  info_queue = Queue()
  for pro_id in range(num_envs):  
      play_process = Process(target=rolloutExperience, args = (model, info_queue, args, pro_id))
      play_process.daemon = True  # sub processes killed when main process finish
      processes.append(play_process)
  
      ### launch update process (single or multiple)
      update_process = Process(target=updateModel, args= (model, info_queue, args, '0'))
      update_process.daemon = True
      processes.append(update_process)
  
      [p.start() for p in processes]
      while all([p.is_alive()for p in processes]):
          pass
  
  ```
  
  
  
* Test the trained model:

  ```python
  from mars.utils.func import LoadYAML2Dict
  from mars.env.import_env import make_env
  from mars.rollout import rollout
  from mars.rl.agents import *
  from mars.rl.agents.multiagent import MultiAgent
  
  ### Load configurations
  yaml_file = 'mars_core/confs/pettingzoo/boxing_v1/pettingzoo_boxing_v1_selfplay'
  args = LoadYAML2Dict(yaml_file, toAttr=True)
  args.multiprocess = False
  
  ## Change/specify some arguments if necessary
  args.test = True  # the test mode will automatically fix all models
  args.render = True
  args.load_model_full_path = 'PATH TO THE TRAINED MODEL' # or use args.load_model_idx
  
  ### Create env
  env = make_env(args)
  
  ### Specify models for each agent
  model1 = eval(args.algorithm)(env, args)
  model2 = eval(args.algorithm)(env, args)
  
  model = MultiAgent(env, [model1, model2], args)
  
  ### Rollout
  rollout(env, model, args)
  ```

* Exploit the trained model:

  ```python
  from mars.env.import_env import make_env
  from mars.rollout import rollout
  from mars.rl.agents import *
  from mars.rl.agents.multiagent import MultiAgent
  from mars.utils.func import get_general_args, get_model_path, get_exploiter
  
  ### Load configurations
  yaml_file = 'mars_core/confs/pettingzoo/boxing_v1/pettingzoo_boxing_v1_selfplay'
  args = LoadYAML2Dict(yaml_file, toAttr=True)
  args.multiprocess = False
  
  ### Change/specify some arguments if necessary
  game_type = 'pettingzoo'
  game = 'boxing_v1'
  method = 'selfplay'
  args.against_baseline = False
  args.test = False
  args.exploit = True
  args.render = False
  load_id = **idx to fill here**
  folder = f'data/model/{load_id}/{game_type}_{game}_{method}/'
  
  args.load_model_full_path = get_model_path(method, folder)
  
  ### Create env
  env = make_env(args)
  print(env)
  
  ### Specify models for each agent
  trained_model = eval(args.algorithm)(env, args)
  # trained_model.fix()  # no longer need to specify here
  
  ### Load exploiter with specified args (just change the previous args)
  args.net_architecture['hidden_dim_list'] = [64, 64, 64, 64]
  exploiter, exploitation_args = get_exploiter('DQN', env, args) # use DQN agent as exploiter
  
  ### Construct multi-agent model
  model = MultiAgent(env, [trained_model, exploiter], exploitation_args)
  
  ### Rollout
  rollout(env, model, exploitation_args, save_id = load_id+'_exploit')
  ```
