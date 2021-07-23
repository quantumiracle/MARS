# A Quick Start

## Single-agent RL

* Train a model on single-agent game, e.g. *CartPole-v1 OpenAI Gym*, using PPO algorithm:

  ```python
  from utils.func import LoadYAML2Dict
  from env.import_env import make_env
  from rollout import rollout
  from rl.algorithm import *
  
  ### Load configurations
  yaml_file = 'mars_core/confs/gym_cartpolev1_ppo'
  args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)
  
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
  from utils.func import LoadYAML2Dict
  from env.import_env import make_env
  from rollout import rollout
  from rl.algorithm import *
  
  ### Load configurations
  yaml_file = 'mars_core/confs/gym_cartpolev1_ppo'
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

* Train a model on two-agent game, e.g. *boxing-v1 PettingZoo*, using self-play with DQN algorithm:

  ```python
  from utils.func import LoadYAML2Dict
  from env.import_env import make_env
  from rollout import rollout
  from rl.algorithm import *
  
  ### Load configurations
  yaml_file = 'mars_core/confs/pettingzoo_boxingv1_selfplay_dqn'
  args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)
  
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
  
* Test the trained model:

  ```python
  ### Load configurations
  yaml_file = 'mars_core/confs/pettingzoo_boxingv1_selfplay_dqn'
  args = LoadYAML2Dict(yaml_file, toAttr=True)
  
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

