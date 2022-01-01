""" Genetic Algorithm """
import torch
import torch.nn as nn

import numpy as np
import copy
from .agent import Agent
from ..common.networks import MLP, CNN
from torch.distributions import Categorical


class GA(Agent):
    """
    GA algorithm
    """
    def __init__(self, env, args):
        super().__init__(env, args)
        self.env = env
        self.args = args
        self.agents = self._init_agents(env, args)
        self.mutation_power = args.algorithm_spec[
            'mutation_power']  #hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf

    def _init_agents(self, env, args):
        agents = []
        for _ in range(args.algorithm_spec['num_agents']):
            agent = MLP(env,
                        args.net_architecture,
                        model_for='discrete_policy').to(self.device)
            for param in agent.parameters():
                param.requires_grad = False
            self._init_weights(agent)
            agents.append(agent)
        return agents

    def _init_weights(self, m):
        if ((type(m) == nn.Linear) | (type(m) == nn.Conv2d)):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.00)

    @property
    def num_agents(self):
        return len(self.agents)

    def choose_action(self, agent_ids, s, Greedy=False):
        a_list = []
        for i, obs in zip(agent_ids, s):
            prob = self.agents[i](torch.from_numpy(obs).unsqueeze(0).float(
            ).to(self.device)).squeeze()  # make sure input state shape is correct
            if Greedy:
                a = torch.argmax(prob, dim=-1).item()
                a_list.append(a)
            else:
                dist = Categorical(prob)
                a = dist.sample().detach().item()
                a_list.append(a)
        return a_list

    def return_children(self, sorted_parent_indexes):
        """ We do not use the elite without noise in our implementation. """
        children_agents = []
        #first take selected parents from sorted_parent_indexes and generate N children
        for i in range(len(self.agents)):
            selected_agent_index = sorted_parent_indexes[np.random.randint(
                len(sorted_parent_indexes))]
            children_agents.append(
                self.mutate(self.agents[selected_agent_index]))

        return children_agents

    def mutate(self, agent):

        child_agent = copy.deepcopy(agent)

        for param in child_agent.parameters():

            if (len(param.shape) == 4):  #weights of Conv2D
                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):
                        for i2 in range(param.shape[2]):
                            for i3 in range(param.shape[3]):

                                param[i0][i1][i2][
                                    i3] += self.mutation_power * np.random.randn(
                                    )

            elif (len(param.shape) == 2):  #weights of linear layer
                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):

                        param[i0][i1] += self.mutation_power * np.random.randn(
                        )

            elif (len(
                    param.shape) == 1):  #biases of linear layer or conv layer
                for i0 in range(param.shape[0]):

                    param[i0] += self.mutation_power * np.random.randn()

        return child_agent

    def save_model(self, path, best_agent_id):
        try:  # for PyTorch >= 1.7 to be compatible with loading models from any lower version
            torch.save(self.agents[best_agent_id].state_dict(), path+'_best_agent', _use_new_zipfile_serialization=False)
        except:
            torch.save(self.agents[best_agent_id].state_dict(), path+'_best_agent')

    def load_model(self, path, eval=True, default_id=0):
        """Load model into one agent with default id """
        self.agents[default_id].load_state_dict(torch.load(path+'_best_agent'))
        if eval:
            self.agents[default_id].eval()