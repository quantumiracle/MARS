import torch
from torch.distributions import Categorical
import numpy as np
import random, copy
from .common.storage import ReplayBuffer, ReservoirBuffer
from .common.rl_utils import choose_optimizer, EpsilonScheduler
from .common.networks import  MLP, CNN
from .common.agent import Agent
from .dqn import DQN, DQNBase
from .equilibrium_solver import * 

class NFSP(Agent):
    """
    Neural Fictitious Self-Play algorithm: https://arxiv.org/pdf/1603.01121.pdf
    """
    def __init__(self, env, args):
        super().__init__(env, args)
        self.rl_agent = DQN(env, args)  # TODO can also use other RL agents
        self.policy = MLP(env.observation_space, env.action_space, args.net_architecture['policy'], model_for='discrete_policy').to(self.device)
        self.replay_buffer = self.rl_agent.buffer
        self.reservoir_buffer = ReservoirBuffer(int(float(args.algorithm_spec['replay_buffer_size'])) )
        self.rl_optimizer = self.rl_agent.optimizer
        self.sl_optimizer = choose_optimizer(args.optimizer)(self.policy.parameters(), lr=float(args.learning_rate))
        self.epsilon_scheduler = self.rl_agent.epsilon_scheduler
        self.schedulers = self.rl_agent.schedulers

        self.eta = float(args.marl_spec['eta'])

    def choose_action(self, state, Greedy=False, epsilon=None):
        self.is_best_response = False
        if random.random() > self.eta:
            prob = self.policy(torch.from_numpy(state).unsqueeze(0).float().to(self.device)).squeeze()  # make sure input state shape is correct
            dist = Categorical(prob)
            action = dist.sample()
            action = action.detach().cpu().numpy()
        else:
            self.is_best_response = True
            action = self.rl_agent.choose_action(state, Greedy, epsilon)
        return action

    def store(self, sample):
        self.replay_buffer.push(sample)
        if self.is_best_response:  # store() needs to be at the same timestep as choose_action()
            self.reservoir_buffer.push(sample)

    @property
    def ready_to_update(self):
        if len(self.replay_buffer) > self.batch_size:
            return True
        else:
            return False

    def update(self):
        ### reinforcement learning (RL) update ###
        rl_loss = self.rl_agent.update()

        ### supervised learning (SL) update ###
        state, action, _, _, _ = self.reservoir_buffer.sample(self.batch_size)

        state = torch.FloatTensor(np.float32(state)).to(self.device)
        action = torch.LongTensor(action).to(self.device)

        probs = self.policy(state)
        probs_with_actions = probs.gather(1, action.unsqueeze(1))
        log_probs = probs_with_actions.log()

        sl_loss = -1 * log_probs.mean()
        
        self.sl_optimizer.zero_grad()
        sl_loss.backward()
        self.sl_optimizer.step()

        return rl_loss + sl_loss.item()

    def save_model(self, path):
        self.rl_agent.save_model(path)
        torch.save(self.policy.state_dict(), path+'_policy')

    def load_model(self, path, eval=True):
        self.rl_agent.load_model(path, eval)
        self.policy.load_state_dict(torch.load(path+'_policy'))

        if eval:
            self.policy.eval()
