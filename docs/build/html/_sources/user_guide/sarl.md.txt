# Single-Agent RL

### RL

The single-agent reinforcement learning is supported in MARS. 

The list of supported algorithms includes: (list here)

### Training

* The followings are required in the main script, for either training/testing/exploitation:

  ```python
  from mars.utils.func import LoadYAML2Dict
  from mars.env.import_env import make_env
  from mars.rollout import rollout
  from mars.rl.agents import *
  from mars.rl.agents.multiagent import MultiAgent
  ```

  

* Typical usage for a single-agent game, e.g. *CartPole-v1 OpenAI Gym*:

  ```python
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


### Testing

* Typical usage for a single-agent game, e.g. *CartPole-v1 OpenAI Gym*:

  ```python
  ### Load configurations
  yaml_file = 'mars/confs/gym_cartpolev1_dqn' #PATH TO YAML
  args = LoadYAML2Dict(yaml_file, toAttr=True)
  print(args)
  
  ## Change/specify some arguments if necessary
  args.test = True  # the test mode will automatically fix all models
  args.render = True
  args.load_model_full_path = 'PATH TO THE TRAINED MODEL' # or use args.load_model_idx
  
  ### Create env
  env = make_env(args)
  print(env)
  
  ### Specify models for each agent
  model = eval(args.algorithm)(env, args)
  
  model = MultiAgent(env, [model], args)
  
  ### Rollout
  rollout(env, model, args)
  ```

  

