# Some Notes

### Environments

* For SlimeVolley environments, if you want to train single agent against the baseline provided by the environment, you need to: 
  1. set *against_baseline* as *True* in either the yaml file or input arguments; then you can use it as the single-agent Gym environment.
  
* For SlimeVolley environments, if you want to train agents with self-play, you need to:

  1. set *against_baseline* as *False* in either the yaml file or input arguments;

  2. do not need to fix any model and just set the configuration yaml file to have the agents to be self-updated (using RL loss) as ['marl_spec']\['trainable_agent_idx'].

* Single- or multiple-agent support:
  * SlimeVolley environments support both single-agent and two-agent games, by setting the *against_baseline* configuration to be *True* or *False* in either case;
  * Openai Gym environments only support single-agent games;
  * PettingZoo environments only support multiple-agent games, unless you can provide some agents as the opponents and set those models with *.fix()* then you can view the learnable agent to play in a single-agent game. 

### RL

The single-agent reinforcement learning is supported in MARS. 

The list of supported algorithms includes: (list here)

### MARL

#### Self-Play

We provide several algorithms, in either reinforcement learning (RL) or evolutionary strategy (ES), with self-play learning mechanism in multi-agent environments, including deep-Q networks (DQN), proximal policy optimization (PPO), genetic algorithm (GA), etc. Therefore, they can be generally classified as **Self-Play + RL** or **Self-Play + ES**.

* Self-Play + RL:

  ```tex
  Champion List:
  Initially, this list contains a random policy agent.
  
  Environment:
  At the beginning of each episode, load the most recent agent archived in the Champion List.
  Set this agent to be the Opponent.
  
  Agent:
  Trains inside the Environment against the Opponent with our choice of RL method.
  Once the performance exceeds some threshold, checkpoint the agent into the Champion List.
  ```

  There are some requirements for an environment to be conveniently learned with self-play + RL method: (1) the environment needs to be symmetric for each agent, including their state space, action space, transition dynamics, random start, etc; (2) to conveniently apply the above mechanism for learning a single model controlling the two sides of the game, the perspectives for different agents needs to be the same, which means, the same model could be applied on each side of the game without modification. The *first* point is obviously satisfied in some games like Go, *SlimeVolley*, and most of the Atari games like *Boxing* and *Pong* , etc. The *second* point is not always available for most multi-agent game although it seems rather like a trivial implementation issue. For example, in all Atari games (OpenAI Gym or PettingZoo), there is only one perspective of observation for all agents in the game, which is the full view of the game (either image or RAM) and contains all information for both the current agent and its opponent. Thus, all agents have the same observation in Atari games by default, which makes the model lack of knowledge it is currently taking charge of. An [issue](https://github.com/PettingZoo-Team/PettingZoo/issues/423) for reference is provided. The direct solution for making a game to  provide same observation perspective for each agent is to transform the perspectives of observation from all agents to the same one. If this is hard or infeasible in practice (e.g. transforming and recoloring the Atari games can take considerable efforts),  an alternative to achieve a similar effect is to add an indicator of the agent to its observation, which is a one-hot vector indicating which agent the sample is collected with, and use all samples to update the model. 

  If both the above two points are satisfied in an environment, we can simply learn one model to control each agent in a game. Moreover, samples from all agents will be symmetric for the model, therefore the model can and should learn from all those samples to maximize its learning efficiency. As an example, with DQN algorithm, we should put samples from all agents into the buffer of the model for it to learn from. Due to this reason, the implementation of self-play in our repository is different from [the previous one](https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ppo_selfplay.py). Since the perspective transformation is provide with in the *SlimeVolley* environment, it can use samples from only one agent to update the model.

  

* Self-Play + GA:

  ```tex
  Create a population of agents with random initial parameters.
  Play a total of N games. For each game:
    Randomly choose two agents in the population and have them play against each other.
    If the game is tied, add a bit of noise to the second agent's parameters.
    Otherwise, replace the loser with a clone of the winner, and add a bit of noise to the clone.
  ```

  

### Configurations

1. In files under the folder `mars_core/confs/`, the configuration entry with value *False* means it is intended to left empty, we do not use *None* since it is not properly recognized as a Python None type but a string type in our file reading process.

2. Training Configuration: 

   The overall training configurations can be specified through either (1) a *yaml* file or (2) using a parser for input arguments.  

   The configurations are classified linguistically into three sets: 

   (1) `env_args` contains arguments for specifying the environments, including the name and type of environments etc; 

   (2) `agent_args` contains arguments for specifying the learning agents, including the algorithm details etc; 

   (3) `train_args` contains arguments for specifying the training details, including network architectures, optimizers, etc. 

Default configurations:

* `env_args`: (dict) # arguments for environments

  * `env_name`: (str) None # name of the environment

  * `env_type`: (str) None # type of the environment, one of [gym, pettingzoo, slimevolley, lasertag]
  * `num_envs`: (int) 1 # number of environments, >1 when using parallel environment sampling
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

### Others

There are a bunch of points worth of taking care of during usage, which are listed as below:

* For greedy-action selection when using stochastic policy (pseudo-policy), like in DQN or PPO, all agent should be **not** be greedy for training; all agents should be greedy for testing; the trained models should be greedy and the exploiters should **not** be greedy for exploitation.

* In iterative best response algorithms, like self-play, fictitious self-play, NXDO, the `score_avg_window` and `selfplay_score_delta` affect the (minimal) interval for storing the model and updating the opponent thus the number of models in the league, the larger the values the fewer the models stored, so choose proper values for each environment to store a proper number of models. 

