# Multi-Agent RL

## MARL algorithms

**Table of contents**

* [Self-Play](#self-play)
* [Fictitious Self-Play](#fictitious-self-play)
* [Neural Fictitious Self-Play](#neural-fictitious-self-play)
* [Policy-Space Response Oracles](#policy-space-response-oracles)

### Self-Play

#### Description

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

  #### Example

  An example to run: 

  ```python
  from utils.func import LoadYAML2Dict
  from env.import_env import make_env
  from rollout import rollout
  from rl.algorithm import *
  
  ### Load configurations
  yaml_file = 'confs/slimevolley_slimevolleyv0_selfplay'
  
  args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)
  
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

  

* Self-Play + ES:

  In the category of self-play with evolutionary strategy algorithms, we implement self-play with genetic algorithm (GA).
  
  ```tex
  Create a population of agents with random initial parameters.
  Play a total of N games. For each game:
    Randomly choose two agents in the population and have them play against each other.
    If the game is tied, add a bit of noise to the second agent's parameters.
    Otherwise, replace the loser with a clone of the winner, and add a bit of noise to the clone.
  ```
  
  #### Example
  
  An example to run:
  
  ```python
  from utils.func import LoadYAML2Dict
  from env.import_env import make_env
  from rollout import rollout
  from rl.algorithm import *
  
  ### Load configurations
  yaml_file = 'confs/slimevolley_slimevolleyv0_selfplay_ga'
  
  args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)
  
  ### Create env
  env = make_env(args)
  
  ### Specify models for each agent
  model = eval(args.algorithm)(env, args)
  
  ### Rollout
  rollout(env, model, args)
  ```
  
  

#### Reference

* [*The repository and tutorials of SlimeVolley environment*](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)*, Hardmaru, et. al*.

* The theory of learning in games. D Fudenberg, F Drew, DK Levine - 1998 

  

### Fictitious Self-Play

#### Description

Instead of playing best response against the opponent's latest strategy as in self-play, fictitious self-play learns the best response against the opponent's historical average strategy and add to its strategy set.

#### Example

 An example to run:

```python
from utils.func import LoadYAML2Dict
from env.import_env import make_env
from rollout import rollout
from rl.algorithm import *

### Load configurations
yaml_file = 'confs/pettingzoo_boxingv1_fsp'

args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)

### Create env
env = make_env(args)

### Specify models for each agent
model1 = eval(args.algorithm)(env, args)
model2 = eval(args.algorithm)(env, args)

model = MultiAgent(env, [model1, model2], args)

### Rollout
rollout(env, model, args)
```

#### Reference

* Iterative solution of games by fictitious play. Brown, George W 1951.

### Neural Fictitious Self-Play

#### Description

Neural Fictitious Self-Play (NFSP) is a generalization of fictitious self-play (which proves to solve Nash equilibrium in two-player zero-sum games and multi-player potential games) with neural networks. In NFSP, each agent is learning through a certain RL algorithm (*e.g.*, DQN, PPO, etc) as well as maintaining a policy network through supervised learning on previous samples, *i.e.*, the average strategy profile, in a sense of the average of the best responses to opponent's historical strategies.

The hyper-parameter ![\eta](https://latex.codecogs.com/svg.latex?\eta) in NFSP is set to choose the action either as the best response to current opponent strategy or from the average strategy that the agent applied as previous best responses.  It can be specified with:

```python
args.marl_spec['eta'] = 0.1
```

In testing, the value of ![\eta](https://latex.codecogs.com/svg.latex?\eta) is automatically set as 0, so that all actions come from the learned average policy. However, as in original [paper](https://arxiv.org/pdf/1603.01121.pdf), there are at least three types of strategies can be derived from NFSP: (1) best response strategy (setting ![\eta](https://latex.codecogs.com/svg.latex?\eta) as 1, also greedy); (2) average strategy (setting ![\eta](https://latex.codecogs.com/svg.latex?\eta) as 0); (3) greedy-average strategy (setting ![\eta](https://latex.codecogs.com/svg.latex?\eta) as 0, but also taking greedy action that maximizes the action value or probability rather than sampling from a probabilistic distribution). Our default choice in implementation for testing is the second one.

#### Example

 An example to run:

```python
from utils.func import LoadYAML2Dict
from env.import_env import make_env
from rollout import rollout
from rl.algorithm import *

### Load configurations
yaml_file = 'confs/pettingzoo_boxingv1_nfsp'

args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)

### Create env
env = make_env(args)

### Specify models for each agent
model1 = eval(args.algorithm)(env, args)
model2 = eval(args.algorithm)(env, args)

model = MultiAgent(env, [model1, model2], args)

### Rollout
rollout(env, model, args)
```

#### Reference

* *[Deep Reinforcement Learning from Self-Play in Imperfect-Information Games](https://arxiv.org/pdf/1603.01121.pdf)*, *Johannes Heinrich and David Silver.*

### Policy-Space Response Oracles

#### Description

Policy-Space Response Oracles (PSRO) is a unified framework for learning in games.

#### Example

 An example to run:

```python
from utils.func import LoadYAML2Dict
from env.import_env import make_env
from rollout import rollout
from rl.algorithm import *

### Load configurations
yaml_file = 'confs/pettingzoo_boxingv1_psro'

args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)

### Create env
env = make_env(args)

### Specify models for each agent
model1 = eval(args.algorithm)(env, args)
model2 = eval(args.algorithm)(env, args)

model = MultiAgent(env, [model1, model2], args)

### Rollout
rollout(env, model, args)
```

#### Reference

* [A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://proceedings.neurips.cc/paper/2017/file/3323fe11e9595c09af38fe67567a9394-Paper.pdf) M Lanctot et. al. 2017

### Nash DQN with Exploiter

#### Description

#### Example

An example to run:

```python
from utils.func import LoadYAML2Dict
from env.import_env import make_env
from rollout import rollout
from rl.algorithm import *

### Load configurations
yaml_file = 'confs/slimevolley_slimevolleyv0_nash_dqn_exploiter'

args = LoadYAML2Dict(yaml_file, toAttr=True, mergeDefault=True)

### Create env
env = make_env(args)

### Specify models for each agent
model1 = eval(args.algorithm)(env, args)
model = MultiAgent(env, [model1], args)

### Rollout
rollout(env, model, args)
```



#### Reference

* 



## General Usage

### Training

* The followings are required in the main script, for either training/testing/exploitation:

  ```python
  from utils.func import LoadYAML2Dict
  from env.import_env import make_env
  from rollout import rollout
  from rl.algorithm import *
  ```

  

* Typical usage for a two-agent game, e.g. *boxing-v1 PettingZoo*, the algorithm details including which MARL algorithm is used are specified in the *yaml* file:

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

* Typical usage for a two-agent game, e.g. *boxing-v1 PettingZoo*, similar as above, the algorithm details including which MARL algorithm is used are specified in the *yaml* file:

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
  
   
