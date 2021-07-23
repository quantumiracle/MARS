# General Description

MARS supports both single-agent RL and multi-agent RL. 

Supported RL algorithms:

* [DQN](https://arxiv.org/abs/1312.5602)

  * You can choose whether to use [Dueling DQN](https://arxiv.org/abs/1511.06581) or not. 

* [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)

  * It is the clipping version in our implementation.

  * You can also choose whether to use [Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438.pdf) or not. 

Supported MARL algorithms:

* Self-Play
* Neural Fictitious Self-Play

Supported environments:

## Configurations

1. In files under the folder `mars_core/confs/`, the configuration entry with value *False* means it is intended to left empty, we do not use *None* since it is not properly recognized as a Python None type but a string type in our file reading process.

2. Training Configuration: 

   The overall training configurations can be specified through either (1) a *yaml* file or (2) using a parser for input arguments.  

   The configurations are classified linguistically into three sets: 

   (1) `env_args`: contains arguments for specifying the environments, including the name and type of environments etc; 

   (2) `agent_args`: contains arguments for specifying the learning agents, including the algorithm details etc; 

   (3) `train_args`: contains arguments for specifying the training details, including network architectures, optimizers, etc. 

Default configurations:

* `env_args`: *(dict)* # arguments for environments
  * `env_name`: *(str)* None # name of the environment
  
  * `env_type`: *(str)* None # type of the environment, one of [gym, pettingzoo, slimevolley, lasertag]
  * `num_envs`: *(int)* 1 # number of environments, >1 when using parallel environment sampling
  * `ram`: (bool) True # whether using RAM observation (instead of using image observation)
  * `seed`: (int) 1122 # random seed
  
* `agent_args`: (dict) # arguments for specifying the learning agent
  * `algorithm`: (str) DQN # the algorithm name, take 'DQN' as an example
  * `algorithm_spec`: (dict) # algorithm specific hyper-parameters
    * `episodic_update`: (bool) False # whether using episodic update or not, if not, take the update per timestep
    * `dueling`: (bool) False # whether using dueling networks in DQN
    * `replay_buffer_size`: (int) 1e5 # size of experience replay buffer
    * `gamma`: (float) 0.99 # discount factor, range [0, 1]
    * `multi_step`: (int) 1 # whether using multi-step reward, i.e. TD(\lambda)
    * `target_update_interval`: (int) 1000 # how many updates are skipped to update the target
    * `eps_start`: (float) 1. # the \epsilon value at the start
    * `eps_final`: (float) 0.01 # the \epsilon value at the end
    * `eps_decay`: (int) 30000  # approximate episodes required to decay from \epsilon value at the start to the value at the end, this needs to be tuned for specific environment

* `train_args`: (dict) # arguments for the training/testing/exploiting process
  * `batch_size`: (int) 32 # training batch size for off-policy algorithms, e.g. DQN
  * `max_episodes`: (int) 10000 # maximal episodes for training
  * `max_steps_per_episode`: (int) 10000 # maximal timesteps per episode
  * `train_start_frame`: (int) 10000 # the number of timesteps skipped before training starts
  * `optimizer`: (str) adam # optimizer type, one of [adam, sgd]
  * `learning_rate`: (float) 1e-4 # learning rate for optimizers
  * `device`: (str) gpu # training device, one of [gpu, cpu]
  * `update_itr`: (int) 1  # iterations of updates per frame, 0~inf; <1 means several steps are skipped per update
  * `log_avg_window`: (int) 20 # average window length in logging
  * `log_interval`: (int) 20  # log print interval 
  * `render`: (bool) False # whether rendering the visualization window
  * `test`: (bool) False # test mode or not
  * `exploit`: (bool) False # exploit mode or not, used for exploiting a trained model
  * `load_model_idx`: (bool/str) False # the index of trained model, default format as 'Timestamp/EpisodeForSavingModel'
  * `load_model_full_path`: (bool/str) False # the complete path to locate the model
  * `net_architecture`: (dict) # network architecture
    * `hidden_dim_list`: (list) [64, 64, 64]  # list of numbers of hidden units
    * `hidden_activation`: (bool/str) ReLU  # activation function for hidden layers, use torch.nn (in Sequential) style rather than torch.nn.functional (in forward)
    * `output_activation`: (bool/str) False # activation function for output layers, False means nan

Note: 

* Different algorithms will have different `algorithm_spec` entries, for example, PPO may use:
  * `algorithm_spec`: (dict)
    * `episodic_update`: (bool) True  # as PPO is on-policy, it uses episodic update instead of update per timestep
    * `gamma`: (float) 0.99 # discount factor
    * `lambda`: (float) 0.95 # hyper-parameter for GAE
    * `eps_clip`: (float) 0.2 # clipping factor 
    * `K_epoch`: (int) 4  # epochs for each update
    * `GAE`: (bool) False  # generalized advantage estimation

## Usage

### Training

* The followings are required in the main script, for either training/testing/exploitation:

  ```python
  from utils.func import LoadYAML2Dict
  from env.import_env import make_env
  from rollout import rollout
  from rl.algorithm import *
  ```

  

* Typical usage for a single-agent game, e.g. *CartPole-v1 OpenAI Gym*:

  ```python
  ### Load configurations
  yaml_file = 'PATH TO YAML'
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

  

* Typical usage for a two-agent game, e.g. *boxing-v1 PettingZoo*:

  ````python
  ### Load configurations
  yaml_file = 'PATH TO YAML'
  args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)
  
  ### Create env
  env = make_env(args)
  print(env)
  
  ### Specify models for each agent
  model1 = eval(args.algorithm)(env, args)
  model2 = eval(args.algorithm)(env, args)
  # model1.fix()  # fix a model if you don't want it to learn
  
  model = MultiAgent(env, [model1, model2], args)
  
  ### Rollout
  rollout(env, model, args)
  
  ````

### Testing

* Typical usage for a single-agent game, e.g. *CartPole-v1 OpenAI Gym*:

  ```python
  ### Load configurations
  yaml_file = 'PATH TO YAML'
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

* Typical usage for a two-agent game, e.g. *boxing-v1 PettingZoo*:

  ```python
  ### Load configurations
  yaml_file = 'PATH TO YAML'
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
  model1 = eval(args.algorithm)(env, args)
  model2 = eval(args.algorithm)(env, args)
  
  model = MultiAgent(env, [model1, model2], args)
  
  ### Rollout
  rollout(env, model, args)
  ```

  

### Exploitation

* When you use SlimeVolley environments and want to exploit a trained model in this type of environment, you need to set the *yaml* file with *against_baseline* as *False* and *exploit* as *True*, so that you can input two models to the *MultiAgent* object, one is the trained model you want to exploit, another one is the exploiter with whatever model you want to use. A typical example would be: 

  ```python
  ### Load configurations
  yaml_file = 'PATH TO YAML'
  args = LoadYAML2Dict(yaml_file, toAttr=True)
  print(args)
  
  ## Change/specify some arguments if necessary
  args.against_baseline = False
  args.exploit = True
  args.load_model_full_path = 'PATH TO THE TRAINED MODEL'
  
  ### Create env
  env = make_env(args)
  print(env)
  
  ### Specify models for each agent
  trained_model = eval(args.algorithm)(env, args)
  exploiter = DQN(env, args)
  trained_model.fix()
  
  model = MultiAgent(env, [trained_model, exploiter], args)
  
  ### Rollout
  rollout(env, model, args)
  ```
  
   

### Model Zoo

​	We provide a zoo of trained agents using default *yaml* configuration files with either single-agent RL or self-play. 
