import torch
import torch.nn as nn
import numpy as np


class Agent(object):
    """
    A standard agent class.
    """
    def __init__(self, env, args):
        super(Agent, self).__init__()
        self.batch_size = args.batch_size
        self.schedulers = []
        if args.device == 'gpu':
            self.device = torch.device("cuda:0")  # TODO
        elif args.device == 'cpu':
            self.device = torch.device("cpu")
        self.not_learnable = False  # whether the model is fixed (not learnable) or not

    def fix(self, ):
        self.not_learnable = True

    def choose_action(self, state, *args):
        pass

    def scheduler_step(self, frame):
        """ Learning rate scheduler, epsilon scheduler, etc"""
        for scheduler in self.schedulers:
            scheduler.step(frame)

    def store(self, *args):
        """ Store a sample for either on-policy or off-policy algorithms."""
        pass

    def update(self):
        """ Update the agent. """
        pass

    def update_target(self, current_model, target_model):
        """
        Update the target model when necessary.
        """
        target_model.load_state_dict(current_model.state_dict())

    def save_model(self, path=None):
        pass

    def load_model(self, path=None):
        pass

    @property
    def ready_to_update(self):
        return True


class MultiAgent(Agent):
    """
    A class containing all agents in a game.

    'not_learnable': 
        Definition:
            The agent is not self-updating using the RL loss, 
            it's either never updated (i.e., 'fixed') or updated as
            a delayed copy of other learnable agents with the MARL learning scheme.
        Utility:
            Agents in the not_learnable_list will not take the store() and update() functions.
    """
    def __init__(self, env, agents, args):
        super(MultiAgent, self).__init__(env, args)
        self.agents = agents
        self.args = args
        self.number_of_agents = len(self.agents)
        self.not_learnable_list = []
        for i, agent in enumerate(agents):
            if args.test or agent.not_learnable or \
                (args.marl_method == 'selfplay' and i != args.marl_spec['trainable_agent_idx']):
                self.not_learnable_list.append(i)
        if len(self.not_learnable_list) < 1:
            prefix = 'No agent'

        else:
            prefix = f'Agents No. {self.not_learnable_list} (index starting from 0)'
        print(prefix + " are not learnable.")

        if args.test or args.exploit:
            if args.load_model_full_path:  # if the full path is specified, it has higher priority than the model index
                model_path = args.load_model_full_path
            else:
                model_path = f"../model/{args.env_type}_{args.env_name}_{args.marl_method}_{args.algorithm}_{args.load_model_idx}"
            self.load_model(model_path)

        if args.marl_method == 'selfplay':
            # since we use self-play (environment is symmetric for each agent), we can use samples from all agents to train one agent
            self.mergeAllSamplesInOne = True
        else:
            self.mergeAllSamplesInOne = False
        # self.mergeAllSamplesInOne = False   # TODO comment out

    def choose_action(self, states):
        """
        states: (agents, envs, state_dim)
        """
        actions = []
        greedy = True if self.args.test else False

        if self.args.marl_method == 'nash':
            # one model for all agents, the model is the first one
            # of self.agents, it directly takes states to generate actions
            actions = self.agents[0].choose_action(states, Greedy=greedy)
        else:
            # each agent will take its corresponding state to generate
            # the corresponding action
            for state, agent in zip(states, self.agents):
                action = agent.choose_action(state, Greedy=greedy)
                actions.append(action)
        return actions

    def scheduler_step(self, frame):
        for agent in self.agents:
            agent.scheduler_step(frame)

    def store(self, sample):
        all_s = []
        if self.args.marl_method == 'nash':
            # one model for all agents, the model is the first one
            # of self.agents, it directly stores the sample constaining all
            self.agents[0].store(sample)

        else:
            for i, agent, *s in zip(np.arange(self.number_of_agents),
                                    self.agents, *sample):
                # when using parallel env, s for each agent can be in shape:
                # [(state1, state2), (action1, action2), (reward1, reward2), (next_state1, next_state2), (done1, done2)]
                # where indices 1,2 represent different envs, thus we need to separate the sample before storing it in each
                # agent, to have the shape like [[state1, action1, reward1, next_state1, done1], [state2, action2, reward2, next_state2, done2]]
                try:
                    s = np.stack(zip(*s))
                except:
                    s = tuple([s])

                # if using self-play, use samples from all players to train the model (due to symmetry)
                if self.mergeAllSamplesInOne:
                    all_s.extend(s)
                elif i not in self.not_learnable_list:  # no need to store samples for not learnable models
                    agent.store(s)

            # store all samples into the trainable agent in self-play
            if self.mergeAllSamplesInOne:
                self.agents[self.args.marl_spec['trainable_agent_idx']].store(
                    all_s)

    def update(self):
        losses = []
        for i, agent in enumerate(self.agents):
            if i not in self.not_learnable_list:
                loss = agent.update()
                losses.append(loss)
            else:
                losses.append(np.nan)
        return losses

    def save_model(self, path=None):
        for agent in self.agents:
            agent.save_model(path)

    def load_model(self, path=None, eval=True):
        for i, agent in enumerate(self.agents):
            if self.args.exploit:
                # in EXPLOIT mode, the exploiter is learnable, thus not loaded from anywhere
                if i in self.not_learnable_list:
                    agent.load_model(path, eval)
                    print(f'Agent No. [{i}] loads model from: ', path)
            else:
                agent.load_model(path, eval)
                print(f'Agent No. [{i}] loads model from: ', path)

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
