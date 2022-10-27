###
# Similar as ppo_gae_continous.py, but change the update function
# to follow the stablebaseline PPO2 (https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/ppo2/ppo2.html#PPO2) and cleanrl (https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)
# it track value of state during sample collection and thus save computation.
###

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# from torch.distributions import Normal
from torch.distributions.normal import Normal
import numpy as np
import wandb
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import argparse
import random

#Hyperparameters
learning_rate = 3e-4
gamma         = 0.99
lmbda         = 0.95
eps_clip      = 0.2
batch_size    = 2048
mini_batch    = int(batch_size//32)
K_epoch       = 10
T_horizon     = 10000
n_epis        = 10000
vf_coef       = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default='test_recall',
        help="the name of this experiment")
    parser.add_argument('--wandb_activate', type=bool, default=False, help='whether wandb')
    parser.add_argument("--wandb_entity", type=str, default=None,
        help="the entity (team) of wandb's project")   
    parser.add_argument('--recall', type=bool, default=False, help='whether recall')
    args = parser.parse_args()
    print(args)
    return args

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super().__init__()
        self.mean = nn.Sequential(
            layer_init(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_actions), std=0.01),
        )
        self.logstd = nn.Parameter(torch.zeros(1, num_actions))

    def forward(self, x):
        action_mean = self.mean(x)
        action_logstd = self.logstd.expand_as(action_mean)

        return action_mean.squeeze(), action_logstd.squeeze()

class Critic(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.model = nn.Sequential(
            layer_init(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, x):
        return self.model(x)


class PPO():
    def __init__(self, num_inputs, num_actions, hidden_size):
        self.data = deque(maxlen=batch_size)  # a ring buffer
        self.max_grad_norm = 0.5
        self.v_loss_clip = True

        self.critic = Critic(num_inputs).to(device)
        self.actor = Actor(num_inputs, num_actions).to(device)

        self.parameters = list(self.critic.parameters()) + list(self.actor.parameters())
        self.optimizer = optim.Adam(self.parameters, lr=learning_rate, eps=1e-5)

    def record_model(self,):
        self.previous_actor = self.actor.state_dict()

    def recall_model(self,):
        for param in self.actor.state_dict().keys():
            self.actor.state_dict()[param].copy_(self.previous_actor[param].data)

    def pi(self, x):
        return self.actor(x)
    
    def v(self, x):
        return self.critic(x)
      
    def get_action_and_value(self, x, action=None):
        mean, log_std = self.pi(x)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        if action is None:
            action = normal.sample()
        log_prob = normal.log_prob(action).sum(-1)
        value =  self.v(x)
        return action.cpu().numpy(), log_prob, value

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self,):
        s, a, r, s_prime, logprob_a, v, done_mask = zip(*self.data)
        s,a,r,s_prime,logprob_a,v,done_mask = torch.tensor(np.array(s), dtype=torch.float), torch.tensor(np.array(a)), \
                                          torch.tensor(np.array(r), dtype=torch.float).unsqueeze(-1), torch.tensor(np.array(s_prime), dtype=torch.float), \
                                        torch.tensor(logprob_a).unsqueeze(-1), torch.tensor(v).unsqueeze(-1), torch.tensor(np.array(done_mask), dtype=torch.float).unsqueeze(-1)
        return s.to(device), a.to(device), r.to(device), s_prime.to(device), done_mask.to(device), logprob_a.to(device), v.to(device)
        
    def train_net(self):
        s, a, r, s_prime, done_mask, logprob_a, v = self.make_batch()
        with torch.no_grad():
            advantage = torch.zeros_like(r).to(device)
            lastgaelam = 0
            for t in reversed(range(s.shape[0])):
                if done_mask[t] or t == s.shape[0]-1:
                    nextvalues = self.v(s_prime[t])
                else:
                    nextvalues = v[t+1]
                delta = r[t] + gamma * nextvalues * (1.0-done_mask[t]) - v[t]
                advantage[t] = lastgaelam = delta + gamma * lmbda * lastgaelam * (1.0-done_mask[t])
            assert advantage.shape == v.shape
            td_target = advantage + v

        b_inds = np.arange(batch_size)
        for epoch in range(K_epoch):    
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, mini_batch):
                end = start + mini_batch
                minibatch_idx = b_inds[start:end]

                bs, ba, blogprob_a, bv = s[minibatch_idx], a[minibatch_idx], logprob_a[minibatch_idx].reshape(-1), v[minibatch_idx].reshape(-1)
                badvantage, btd_target = advantage[minibatch_idx].reshape(-1), td_target[minibatch_idx].reshape(-1)

                if not torch.isnan(badvantage.std()):
                    badvantage = (badvantage - badvantage.mean()) / (badvantage.std() + 1e-8) 
        
                _, newlogprob_a, new_vs = self.get_action_and_value(bs, ba)
                new_vs = new_vs.reshape(-1)
                ratio = torch.exp(newlogprob_a - blogprob_a)  # a/b == exp(log(a)-log(b))
                surr1 = -ratio * badvantage
                surr2 = -torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * badvantage
                policy_loss = torch.max(surr1, surr2).mean()

                if self.v_loss_clip: # clipped value loss
                    v_clipped = bv + torch.clamp(new_vs - bv, -eps_clip, eps_clip)
                    value_loss_clipped = (v_clipped - btd_target) ** 2
                    value_loss_unclipped = (new_vs - btd_target) ** 2
                    value_loss_max = torch.max(value_loss_unclipped, value_loss_clipped)
                    value_loss =  0.5 * value_loss_max.mean()
                else:
                    value_loss = F.smooth_l1_loss(new_vs, btd_target)

                loss = policy_loss + vf_coef * value_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
                self.optimizer.step()
        
def main():
    args = parse_args()
    env_id = 2
    seed = 1
    env_name = ['HalfCheetah-v2', 'Ant-v2', 'Hopper-v2'][env_id]
    threshold = [400, 80, 20][env_id]
    env = gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    state_dim = env.observation_space.shape[0]
    action_dim =  env.action_space.shape[0]
    hidden_dim = 64
    model = PPO(state_dim, action_dim, hidden_dim)
    score = 0.0
    print_interval = 1
    step = 0
    update = 1
    avg_epi_r = []
    last_avg_epi_r = -10000  # small enough
    eff_update = 1

    if args.wandb_activate:
        wandb.init(
            project=args.run,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=args.run+f'_{env_name}:'+str(args.recall),
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{args.run}_{env_name}:{str(args.recall)}")

    for n_epi in range(n_epis):
        s = env.reset()
        done = False
        epi_r = 0.
        ## learning rate schedule
        # frac = 1.0 - (n_epi - 1.0) / n_epis
        # lrnow = frac * learning_rate
        # model.optimizer.param_groups[0]["lr"] = lrnow

        # while not done:
        for t in range(T_horizon):
            step += 1
            with torch.no_grad():
                a, logprob, v = model.get_action_and_value(torch.from_numpy(s).float().unsqueeze(0).to(device))
            s_prime, r, done, info = env.step(a)
            # env.render()

            model.put_data((s, a, r, s_prime, logprob, v.squeeze(-1), done))
            s = s_prime

            score += r
            epi_r += r

            if step % batch_size == 0 and step > 0:
                model.train_net()
                update += 1
                eff_update = update

            if done:
                break
            # if 'episode' in info.keys():
            #     print(f"Global steps: {step}, score: {info['episode']['r']}")

        if n_epi%print_interval==0 and n_epi!=0:
            print("Global steps: {}, # of episode :{}, avg score : {:.1f}".format(step, n_epi, score/print_interval))
            writer.add_scalar("charts/episodic_return", score/print_interval, n_epi)
            writer.add_scalar("charts/episodic_length", t, n_epi)
            writer.add_scalar("charts/update", update, n_epi)
            writer.add_scalar("charts/effective_update", eff_update, n_epi)
            writer.add_scalar("charts/recall_rate", float(update-eff_update)/update, n_epi)

            score = 0.0

        avg_epi_r.append(epi_r)

    env.close()

if __name__ == '__main__':
    main()