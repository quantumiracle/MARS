# Some Notes

### Environments

* For SlimeVolley environments, if you want to train single agent against the baseline provided by the environment, you need to: 
  1. set *against_baseline* as *True* in either the yaml file or input arguments; then you can use it as the single-agent Gym environment.
  2. ~~instantiate two agent models in `run.py` and fix the first one, so that only the second model is learnable since in single-agent SlimeVolley environments the learnable one is set as the second one by default.~~

* For SlimeVolley environments, if you want to train agents with self-play, you need to:

  1. set *against_baseline* as *False* in either the yaml file or input arguments;

  2. do not need to fix any model and just set the configuration yaml file to have the agents to be self-updated (using RL loss) as ['marl_spec']\['trainable_agent_idx'].

* Single- or multiple-agent support:
  * SlimeVolley environments support both single-agent and two-agent games, by setting the *against_baseline* configuration to be *True* or *False* in either case;
  * Openai Gym environments only support single-agent games;
  * PettingZoo environments only support multiple-agent games, unless you can provide some agents as the opponents and set those models with *.fix()* then you can view the learnable agent to play in a single-agent game. 

### Configurations

* In files under the folder`mars_core/confs/`, the configuration entry with value *False* means it is intended to left empty, we do not use *None* since it is not properly recognized as a Python None type but a string type in our file reading process.

* Training Configuration: 

  The overall training configurations can be specified through either (1) a *yaml* file or (2) using a parser for input arguments.  

  The configurations are classified linguistically into three parts: (1) `env_args` contains arguments for specifying the environments, including the name and type of environments etc; (2) `agent_args` contains arguments for specifying the learning agents, including the algorithm details etc; (3) `train_args` contains arguments for specifying the training details, including network architectures, optimizers, etc. 

### Training



### Exploitation

* When you use SlimeVolley environments and want to exploit a trained model in this type of environment, you need to set the *yaml* file with *against_baseline* as *False*, so that you can input two models to the *MultiAgent* object, one is the trained model you want to exploit, another one is the exploiter with whatever model you want to use. A typical example would be: 

  ```python
  ### Specify models for each agent
  trained_model = eval(args.algorithm)(env, args)
  exploiter = DQN(env, args)
  trained_model.fix()
  
  model = MultiAgent(env, [trained_model, exploiter], args)
  
  ### Rollout
  rollout(env, model, args)
  ```

   

### Model Zoo

