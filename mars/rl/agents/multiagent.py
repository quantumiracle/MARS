import torch
import torch.nn as nn
import numpy as np
from mars.utils.typing import List, Union, StateType, ActionType, SampleType, SamplesType, ConfigurationDict
from mars.env.import_env import make_env
from .agent import Agent
from .dqn import DQN
from .ppo import PPO
from mars.marl import MetaLearner
from mars.utils.common import SelfplayBasedMethods, MetaStrategyMethods, MetaStepMethods, NashBasedMethods

class MultiAgent(Agent):
    """A class containing all agents in a game.

    The 'not_learnable' list is a list containing all not self-learnable agents.
    
        Definition of 'not_learnable': The agent is not self-updating using the RL loss, it's either\
        never updated (i.e., 'fixed') or updated as a delayed copy of other learnable \
        agents with the MARL learning scheme.

        What happen if an agent is 'not_learnable': Agents in the not_learnable_list will not take the store() and update() functions.

    Other things to notice:
        
        * In testing or exploiting modes, if `load_model_full_path` is given, it is preferred over `load_model_idx` to load the existing model.

        * `mergeAllSamplesInOne` is True only for Self-Play method, it means samples from all agents can be used for updating each agent due to symmetry.

        * In Nash method and exploitation mode, the first agent in `agents` list needs to be the one to be exploited (pre-trained).

    :param env: env object
    :type env: object
    :param agents: list of agent models
    :type agents: list
    :param args: all arguments
    :type args: dict
    """    
    def __init__(self, env, agents, args: ConfigurationDict, *Args, **Kwargs):
        """Initialization
        """        
        super(MultiAgent, self).__init__(env, args)
        self.Args = Args
        self.Kwargs = Kwargs
        self.agents = agents
        self.args = args
        self.number_of_agents = len(self.agents)
        self.not_learnable_list = []
        self.mergeAllSamplesInOne = False
        if args.exploit:
            try:
                self.idx_exploited_model = args['idx_exploited_model']  # model to be exploited
            except: 
                print('Error: no exploited model index given!')
                self.idx_exploited_model = 0

        ## Below is a complicated filter process for configuring multiagent class ##
        for i, agent in enumerate(agents):
            if args.test:
                agent.fix()
                self.not_learnable_list.append(i)
            elif args.exploit:
                if i == self.idx_exploited_model:  # fix the model to be exploited
                    agent.fix()
                if agent.not_learnable:
                    self.not_learnable_list.append(i)
            else: # training mode
                if i>0 and args.marl_method in NashBasedMethods: # only one common model is trained
                    self.not_learnable_list.append(i)
                if agent.not_learnable or (args.marl_method in MetaStepMethods and i != args.marl_spec['trainable_agent_idx']):  # psro is special, fixed one side at beginning
                    self.not_learnable_list.append(i)
        if len(self.not_learnable_list) < 1:
            prefix = 'No agent'

        else:
            prefix = f'Agents No. {self.not_learnable_list} (index starting from 0)'
        print(prefix + " are not learnable.")

        if args.test or args.exploit:
            if args.marl_method in MetaStrategyMethods:  
                meta_learner = MetaLearner()  # meta-learner as a single agent to test/exploit
                assert self.idx_exploited_model in self.not_learnable_list # 0 is the model to test/exploit
                meta_learner.load_model(self.agents[self.idx_exploited_model], path=args.load_model_full_path)  
                self.agents[self.idx_exploited_model] = meta_learner
                self.meta_learner = self.agents[self.idx_exploited_model]

            else:
                if args.load_model_full_path:  # if the full path is specified, it has higher priority than the model index
                    model_path = args.load_model_full_path
                else:
                    model_path = f"../model/{args.env_type}_{args.env_name}_{args.marl_method}_{args.algorithm}_{args.load_model_idx}"
                self.load_model(model_path)

        else:  # training mode
            if args.marl_method in SelfplayBasedMethods:  
                # since we use self-play (environment is symmetric for each agent), we can use samples from all agents to train one agent
                self.mergeAllSamplesInOne = True                

        if self.args.marl_method in NashBasedMethods and self.args.exploit:
            assert self.idx_exploited_model in self.not_learnable_list  # the first agent must be the model to be exploited in Nash method, since the first agent stores samples 

    def _choose_greedy(self, )->List[bool]:
        """
        Determine whether each agent should choose greedy actions under different cases.
        
        :return: list of bools indicating greedy action or not for all agents.
        "rtype: list[bool]
        """
        greedy_list = self.number_of_agents*[False]
        if self.args.test:
            greedy_list = self.number_of_agents*[True]
        else:  
            # For train and expoit.
            # 1. train:
            # not learnable agent means fixed opponent;
            # 2. exploit:
            # not learnable agent means model to be exploited;
            # both of the above cases should have greedy action.
            for i in self.not_learnable_list:
                greedy_list[i] = True

        return greedy_list
    
    def choose_action(
        self, 
        states: Union[List[StateType], List[List[StateType]]],
        greedy_list = []
        ) -> Union[List[ActionType], List[List[ActionType]]]:
        """Choose actions from given states/observations.
        Shape of states:  (agents, envs, state_dim)

        :param states: observations for all agents, shape: (agents, envs, action_dim)
        :type states: np.ndarray or list
        :return: actions for all agents, shape: (agents, envs, action_dim)
        :rtype: list
        """        
        actions = []
        if len(greedy_list)==0: # otherwise using the passed in greedy list
            greedy_list = self._choose_greedy()

        if self.args.marl_method in NashBasedMethods: 
            if self.args.exploit:  # in exploitation mode, nash policy only control one agent
                for i, (state, agent, greedy) in enumerate(zip(states, self.agents, greedy_list)):
                    if i == self.idx_exploited_model:  # the first agent must be the model to be exploited
                        if self.args.marl_spec['global_state']:  # use concatenated observation from both agents
                            nash_actions = self.agents[i].choose_action(states, Greedy=greedy)  # nash_actions contain all agents
                        else:  # only use the observation from the first agent
                            nash_actions = self.agents[i].choose_action(np.expand_dims(state, 0), Greedy=greedy)  # (envs, state_dim) to (1, envs, state_dim), 1 for one-side observation
                        actions.append(nash_actions[i])
                    else:
                        action = agent.choose_action(state, Greedy=greedy)
                        actions.append(action)
            else:
                # in training/testing mode, one model for all agents, the model is the first one
                # of self.agents, it directly takes states to generate actions
                if self.args.marl_spec['global_state']:
                    actions = self.agents[0].choose_action(states, Greedy=greedy_list[0])
                else:
                    actions = self.agents[0].choose_action(np.expand_dims(states[0], 0), Greedy=greedy_list[0])  # keep the dim unchanged
        else:
            # each agent will take its corresponding state to generate
            # the corresponding action
            for state, agent, greedy in zip(states, self.agents, greedy_list):
                action = agent.choose_action(state, Greedy=greedy)
                actions.append(action)

        return actions

    def scheduler_step(self, frame: int) -> None:
        for i, agent in enumerate(self.agents):
            if i not in self.not_learnable_list:
                agent.scheduler_step(frame)

    def store(
        self, 
        samples: SamplesType,
        ) -> None:
        """Store the samples into each agent.
        The input samples is for all agents and all environments, we separate and reshape it before calling store() for each agent object. 
        For each item in samples (like `states`, `actions`), it has the shape: (`agents`, `envs`, `sample_dim`).

        Generally, we want each agent to take its samples by indexing the first dimension, therefore (`envs`, `a complete sample`), `a complete sample`
        looks like [state, action, reward, next_state, (other_info, ) done].
        Specifically, for different methods (Self-Play, NFSP, Nash, etc) and different modes (train, test, exploit), the process will be different.
        
            1. If using Nash agent, take the `states` item in samples as an example, it has shape (`agents`, `envs`, `sample_dim`) for parallel environments
            and (`agents`, `sample_dim`) for single environment, we transfer it to be (`envs`, `agents*sample_dim`) for the Nash agent in training and testing modes.
            For exploitation mode, since the Nash agent is fixed and only provide actions for one agent, it is `not_learnable` and does not store any samples, only its
            opponent exploiter store samples in a standard way.

            2. If using Self-Play for training/testing, samples from all agents can be stored in one model for learning (due to the symmetry),
            `mergeAllSamplesInOne` is set to be True. Each agent will take in samples in shape (`envs`, `agents`, `a complete sample`).

            3. If using other methods like single-agent RL or Neural Fictitious Self-Play (NFSP) for training/testing, each agent will only takes their own samples,
            thus `mergeAllSamplesInOne` is False, and the shape of samples for each agent is (`envs`, `a complete sample`).

        :param samples: list of samples from different environment (if using parallel envs) and
         for different agents, it consists of [states, actions, rewards, next_states, (other_infos,) dones].
        :type samples: list
        """
        all_s = []
        if self.args.marl_method != 'nash_ppo' and self.args.marl_method in NashBasedMethods and not self.args.exploit:
            # 'states' (agents, envs, state_dim) -> (envs, agents, state_dim), similar for 'actions', 'rewards' take the first one in all agents,
            # if np.all(d) is True, the 'states' and 'rewards' will be absent for some environments, so remove such sample.
            [states, actions, rewards, next_states, dones] = samples
            try:  # when num_envs > 1. 
                if self.args.marl_spec['global_state']:  # use concatenated observation from both agents
                    samples = [[states[:, j].reshape(-1), actions[:, j].reshape(-1), rewards[0, j], next_states[:, j].reshape(-1), np.any(d)] for j, d in enumerate(np.array(dones).T)]
                else:  # only use the observation from the first agent (assume the symmetry in the game and the single state contains the full information: speed up learning!)
                    samples = [[states[0, j], actions[:, j].reshape(-1), rewards[0, j], next_states[0, j], np.any(d)] for j, d in enumerate(np.array(dones).T)]
            except:  # when num_envs = 1 
                if self.args.marl_spec['global_state']: 
                    samples = [[np.array(states).reshape(-1), actions, rewards[0], np.array(next_states).reshape(-1), np.all(dones)]]  # done for both player
                else:
                    samples = [[np.array(states[0]), actions, rewards[0], np.array(next_states[0]), np.all(dones)]]
           
            # one model for all agents, the model is the first one
            # of self.agents, it directly stores the sample constaining all
            for agent in self.agents[:1]: # actually only one agent in list
                agent.store(samples) 

        elif self.args.marl_method == 'nash_ppo' and not self.args.exploit:
            [states, actions, rewards, next_states, logprobs, dones] = samples
            assert self.args.marl_spec['global_state'],  'Error: Nash PPO should use global state'# this has to be true for Nash PPO
            if self.args.num_envs > 1:  # Used when num_envs > 1. 
                if self.args.ram: # shape of state: (agents, envs, obs_dim)
                    samples = [[states[:, j].reshape(-1), actions[:, j].reshape(-1), rewards[:, j], next_states[:, j].reshape(-1), logprobs[:, j].reshape(-1), np.any(d)] for j, d in enumerate(np.array(dones).T)]
                else: # shape of state: (agents, envs, C, H, W)
                    samples = [[np.array(states)[:, j], actions[:, j], rewards[:, j], np.array(next_states)[:, j], np.array(logprobs[:, j]), np.any(d)] for j, d in enumerate(np.array(dones).T)]

            else:  # when num_envs = 1 
                if self.args.ram:
                    samples = [[np.array(states).reshape(-1), actions, rewards, np.array(next_states).reshape(-1), np.array(logprobs).reshape(-1), np.all(dones)]]
                else:  # TODO
                    samples = [[np.array(states), actions, rewards, np.array(next_states), np.array(logprobs), np.all(dones)]]
 
            # one model for all agents, the model is the first one
            # of self.agents, it directly stores the sample constaining all
            for agent in self.agents[:1]: # actually only one agent in list
                agent.store(samples) 

        else:
            for i, agent, *s in zip(np.arange(self.number_of_agents),
                                    self.agents, *samples):
                # when using parallel env, s for each agent can be in shape:
                # [(state1, state2), (action1, action2), (reward1, reward2), (next_state1, next_state2), (done1, done2)]
                # where indices 1,2 represent different envs, thus we need to separate the sample before storing it in each
                # agent, to have the shape like [[state1, action1, reward1, next_state1, done1], [state2, action2, reward2, next_state2, done2]]
                try: # when num_envs > 1 
                    s = np.stack(zip(*s))
                except: # when num_envs = 1 
                    s = tuple([s])
                # if using self-play, use samples from all players to train the model (due to symmetry)
                if self.mergeAllSamplesInOne:
                    all_s.extend(s)
                elif i not in self.not_learnable_list:  # no need to store samples for not learnable models
                    agent.store(s)
            # store all samples into the trainable agent in self-play
            if self.mergeAllSamplesInOne:
                self.agents[self.args.marl_spec['trainable_agent_idx']].store(all_s)

    def nan_filter(self, samples):
        """This cannot work if None exists in samples, so only use to check np.nan when None not exists.

        :param samples: [description]
        :type samples: [type]
        :return: [description]
        :rtype: [type]
        """
        valid = True
        for i in range(len(samples)):
            if np.isnan(samples[i]).any():  # any entry in item is nan
                samples[i] = np.ones_like(samples)
                print(f"Invalid nan value exists in {i}-th item of samples.")
                valid = False
                break
            else:
                valid = True
        return valid

    def update(self) -> List[float]:
        losses = []
        infos = []
        for i, agent in enumerate(self.agents):
            if i not in self.not_learnable_list:
                loss, info = agent.update()
                losses.append(loss)
                infos.append(info)
            else:
                losses.append(np.nan)
                infos.append({})
        return losses, infos

    def save_model(self, path: str = None) -> None:
        for idx, agent in enumerate(self.agents):
            agent.save_model(path+f'_{str(idx)}')

    def load_model(self, path: str = None, eval: bool = True) -> None:
            
        for i, agent in enumerate(self.agents):
            if isinstance(path, list): # if pass in a list of paths, each agent takes one path in order
                spec_path = path[i]
            else:
                spec_path = path

            if self.args.exploit:
                # in EXPLOIT mode, the exploiter is learnable, thus not loaded from anywhere
                if i in self.not_learnable_list:
                    try:
                        agent.load_model(spec_path, eval)
                        print(f'Agent No. [{i}] loads model from: ', spec_path)
                    except:
                        print(f'Load model failed for the {i} agent.')
            else:
                agent.load_model(spec_path, eval)
                print(f'Agent No. [{i}] loads model from: ', spec_path)

    @property
    def ready_to_update(self) -> bool:
        """ 
        A function returns whether all learnable agents are ready to be updated, 
        called from the main rollout function.
        """
        if self.args.test:  # no model is updated in test mode
            return False
        else:
            ready_state = []
            for i, agent in enumerate(self.agents):
                if i not in self.not_learnable_list:
                    ready_state.append(agent.ready_to_update)
            if np.all(ready_state):
                return True
            else:
                return False
