import torch
from torch.distributions import Categorical
import numpy as np
import random, copy
from ..common.storage import ReplayBuffer, ReservoirBuffer
from ..common.rl_utils import choose_optimizer, EpsilonScheduler
from ..common.networks import  MLP, CNN, ImpalaCNN, get_model
from .agent import Agent
from .dqn import DQN, DQNBase
from mars.equilibrium_solver import * 

class NFSP(Agent):
    """
    Neural Fictitious Self-Play algorithm: https://arxiv.org/pdf/1603.01121.pdf
    """
    def __init__(self, env, args):
        super().__init__(env, args)
        self.rl_agent = DQN(env, args)  # TODO can also use other RL agents
        if isinstance(env.observation_space, list):  # when using parallel envs
            observation_space = env.observation_space[0]
        else:
            observation_space = env.observation_space

        if len(observation_space.shape) <= 1: # not image
            self.policy = get_model('mlp')(env.observation_space, env.action_space, args.net_architecture['policy'], model_for='discrete_policy').to(self.device)
        else:
            self.policy = get_model('impala_cnn')(env.observation_space, env.action_space, args.net_architecture['policy'], model_for='discrete_policy').to(self.device)

        if args.multiprocess:
            self.rl_agent.share_memory()
            self.policy.share_memory()
            self.replay_buffer = args.add_components['replay_buffer']
            self.reservoir_buffer = args.add_components['reservoir_buffer']
        else:
            self.replay_buffer = self.rl_agent.buffer
            self.reservoir_buffer = ReservoirBuffer(int(float(args.algorithm_spec['replay_buffer_size'])) )
        
        self.rl_optimizer = self.rl_agent.optimizer
        self.sl_optimizer = choose_optimizer(args.optimizer)(self.policy.parameters(), lr=float(args.learning_rate))
        self.schedulers = self.rl_agent.schedulers

        self.eta = 0. if args.test else float(args.marl_spec['eta'])  # in test mode, only use average policy
        self.args = args

    def choose_action(self, state, Greedy=False, epsilon=None):
        self.is_best_response = False
        if random.random() > self.eta:
            if self.args.num_envs == 1:
                prob = self.policy(torch.from_numpy(state).unsqueeze(0).float().to(self.device)).squeeze()  # make sure input state shape is correct
            else:
                prob = self.policy(torch.from_numpy(state).float().to(self.device)).squeeze()  # make sure input state shape is correct

            dist = Categorical(prob)
            action = dist.sample()
            try:
                action = action.detach().item()
            except:
                action = action.detach().cpu().numpy()  # when parallel envs
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
        try:  # for PyTorch >= 1.7 to be compatible with loading models from any lower version
            torch.save(self.policy.state_dict(), path+'_policy', _use_new_zipfile_serialization=False)
        except:
            torch.save(self.policy.state_dict(), path+'_policy')

    def load_model(self, path, eval=True):
        self.rl_agent.load_model(path, eval)
        self.policy.load_state_dict(torch.load(path+'_policy'))

        if eval:
            self.policy.eval()
