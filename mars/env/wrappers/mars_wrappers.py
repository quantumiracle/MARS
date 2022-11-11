import random
import gym
import numpy as np

class PettingzooClassicWrapper():
    """ A class for PettingZoo classic games.

    :param env: game environment
    :type env: object
    :param observation_mask: mask the observation to be anyvalue, defaults to None.
    :type observation_mask: int or None, optional
    """   
    def __init__(self, env, observation_mask=None):  
        super(PettingzooClassicWrapper, self).__init__()
        self.env = env
        self.observation_mask = observation_mask
        self.action_spaces = self.env.action_spaces
        self.action_space = list(self.action_spaces.values())[0]
        self.agents = list(self.action_spaces.keys())
        self.num_agents = len(self.agents)      
        # for rps_v1, discrete to box, fake space
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.uint8)
        self.observation_spaces = {a:self.observation_space for a in self.agents}

        # for holdem
        # obs_space = self.env.observation_spaces.values()
        # obs_len = obs_space.shape[0]-action_space.n
        # self.observation_spaces = Box(shape=(obs_len,),low=obs_space.low[:obs_len],high=obs_space.high[:obs_len])

    @property
    def spec(self):
        return self.env.spec

    def reset(self, observation=None):
        obs = self.env.reset()
        if self.observation_mask is not None:
            return {a: self.observation_mask for a in self.agents}
        else:
            return obs

    def seed(self, seed):
        try:
            self.env.seed(seed)
        except:
            self.env.reset(seed=seed) # gym update for seeding
        np.random.seed(seed)

    def render(self,):
        return self.env.render()

    def close(self):
        self.env.close()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self.observation_mask is not None:  # masked observation with a certain value
            observation = {a:self.observation_mask for a in observation.keys()}
        return observation, reward, done, info

class PettingzooClassic_Iterate2Parallel():
    """ A class transforms the iterative environment in PettingZoo
    to parallel version, i.e., from iterative rollout among each player
    to their simultaneuous moves at the same step.

    :param env: game environment
    :type env: object
    :param observation_mask: mask the observation to be anyvalue, defaults to None.
    :type observation_mask: int or None, optional
    """ 
    def __init__(self, env, observation_mask=None):         
        super(PettingzooClassic_Iterate2Parallel, self).__init__()
        self.env = env
        self.observation_mask = observation_mask
        self.action_spaces = self.env.action_spaces
        self.action_space = list(self.action_spaces.values())[0]
        self.agents = list(self.action_spaces.keys())
        self.num_agents = len(self.agents)
        self.observation_space = list(self.env.observation_spaces.values())[0]['observation']
        self.observation_spaces = {a:self.observation_space for a in self.agents}

    @property
    def spec(self):
        return self.env.spec

    def reset(self, observation=None):
        obs = self.env.reset()
        if self.observation_mask is not None:
            return {a: self.observation_mask for a in self.agents}
        else:
            if obs is None:
                return {a:np.zeros(self.observation_space.shape[0]) for a in self.agents} # return zeros
            else:
                return {a: obs[a]['observation'] for a in self.agents}

    def seed(self, seed):
        try:
            self.env.seed(seed)
        except:
            self.env.reset(seed=seed) # gym update for seeding
        np.random.seed(seed)

    def render(self,):
        return self.env.render()

    def close(self):
        self.env.close()

    def step(self, action_dict):
        obs_dict, reward_dict, done_dict, info_dict = {}, {}, {}, {}
        for agent, action in action_dict.items():
            observation, reward, done, info = self.env.last()  # observation is the last action of opponent
            valid_actions = np.where(observation['action_mask'])[0]
            if done: 
                action = None  # for classic game: if one player done (requires to set action None), another is not, it causes problem when using parallel API
            elif action not in valid_actions:
                action = random.choice(valid_actions) # randomly select a valid action
            self.env.step(action)
            if self.observation_mask is not None:  # masked zero/ones observation
                obs_dict[agent] = self.observation_mask
            else:
                obs_dict[agent] = observation['observation']  # observation contains {'observation': ..., 'action_mask': ...}
            reward_dict[agent] = self.env.rewards[agent]
            done_dict[agent] = self.env.dones[agent]  # the returned done from env.last() does not work
            info_dict[agent] = info

        return obs_dict, reward_dict, done_dict, info_dict


class RoboSumoWrapper():
    """ Wrap robosumo environments """
    def __init__(self, env, mode):
        super(RoboSumoWrapper, self).__init__()
        self.env = env
        self.agents = ['first_0', 'second_0']
        self.num_agents = len(self.agents)
        self.observation_space = self.env.observation_space[0]
        self.observation_spaces = {name: self.observation_space for name in self.agents}
        self.action_space = self.env.action_space[0]
        self.action_spaces = {name: self.action_space for name in self.agents}
        self.metadata = env.metadata
        self.render_mode = mode

    @property
    def spec(self):
        return self.env.spec

    def reset(self):
        obs = self.env.reset()
        return obs

    def seed(self, seed):
        self.env.seed(seed)
        np.random.seed(seed)

    def render(self, mode='human'):
        mode = self.render_mode  # force the mode here
        return self.env.render(mode)

    def step(self, actions):
        actions = [a.squeeze() for a in actions]
        try:
            obs, reward, done, info = self.env.step(actions)
        except:
            print(f'Action exception in Mujoco: {actions}')
            obs, reward, done, info = self.env.step(np.zeros_like(actions))
        return obs, reward, done, info

    def close(self):
        self.env.close()

class ZeroSumWrapper():
    """ Filter non-zero-sum rewards to be zero-sum """
    def __init__(self, env):
        super(ZeroSumWrapper, self).__init__()
        self.env = env
        self.agents = env.agents
        self.num_agents = env.num_agents
        self.observation_space = self.env.observation_space
        self.observation_spaces = self.env.observation_spaces
        self.action_space = self.env.action_space
        self.action_spaces = self.env.action_spaces
        self.metadata = env.metadata

    @property
    def unwrapped(self,):
        return self.env

    @property
    def spec(self):
        return self.env.spec

    def _zerosum_filter(self, r):
        ## zero-sum filter: 
        # added for making non-zero sum game to be zero-sum, e.g. tennis_v2, pong_v3
        r = np.array(r)
        if np.sum(r) != 0:
            nonzero_idx = np.nonzero(r)[0][0]
            r[1-nonzero_idx] = -r[nonzero_idx]
        return r

    def reset(self):
        obs = self.env.reset()
        return obs

    def seed(self, seed):
        self.env.seed(seed)

    def render(self, mode='rgb_array'):
        return self.env.render(mode)

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        return obs, self._zerosum_filter(reward), done, info

    def close(self):
        self.env.close()

def zero_sum_reward_filer(r):
    ## zero-sum filter: 
    # added for making non-zero sum game to be zero-sum, e.g. tennis_v2, pong_v3
    r = np.array(r)
    if np.sum(r) != 0:
        nonzero_idx = np.nonzero(r)[0][0]
        r[1-nonzero_idx] = -r[nonzero_idx]
    return r    



class SSVecWrapper():
    """ Wrap after supersuit vector env """
    def __init__(self, env):
        super(SSVecWrapper, self).__init__()
        self.env = env
        if len(env.observation_space.shape) > 1: # image, obs space: (H, W, C) -> (C, H, W)
            old_shape = env.observation_space.shape
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8)
            self.obs_type = 'rgb_image'
        else:
            self.observation_space = env.observation_space
            self.obs_type = 'ram'

        self.action_space = env.action_space
        self.num_agents = env.num_agents
        self.agents = env.agents
        self.true_num_envs = self.env.num_envs//self.env.num_agents
        self.num_envs = self.true_num_envs
    
    @property
    def spec(self):
        return self.env.spec

    def reset(self):
        obs = self.env.reset() 
        if len(self.observation_space.shape) >= 3:
            obs = np.moveaxis(obs, -1, 1) # (N, H, W, C) -> (N, C, H, W)
            obs = obs.reshape(self.true_num_envs, self.env.num_agents, obs.shape[-3], obs.shape[-2], obs.shape[-1])
        else:
            obs = obs.reshape(self.true_num_envs, self.env.num_agents, -1)
        return obs

    def seed(self, seed):
        self.env.seed(seed)

    def render(self, mode='rgb_array'):
        return self.env.render(mode)

    def step(self, actions):
        actions = actions.reshape(-1)
        obs, reward, done, info = self.env.step(actions)
        if len(self.observation_space.shape) >= 3:
            obs = np.moveaxis(obs, -1, 1) # (N, H, W, C) -> (N, C, H, W)
            obs = obs.reshape(self.true_num_envs, self.env.num_agents, obs.shape[-3], obs.shape[-2], obs.shape[-1])
        else:
            obs = obs.reshape(self.true_num_envs, self.env.num_agents, -1)
        reward = reward.reshape(self.true_num_envs, self.env.num_agents)
        done = done.reshape(self.true_num_envs, self.env.num_agents)
        info = [info[:self.true_num_envs], info[self.true_num_envs:]]
        return obs, reward, done, info

    def close(self):
        self.env.close()


class Gym2AgentWrapper():
    """ Wrap single agent OpenAI gym game to be multi-agent version """
    def __init__(self, env):
        super(Gym2AgentWrapper, self).__init__()
        self.env = env
        self.agents = ['first_0']
        self.num_agents = len(self.agents)
        self.observation_space = self.env.observation_space
        self.observation_spaces = {name: self.env.observation_space for name in self.agents}
        self.action_space = self.env.action_space
        self.action_spaces = {name: self.action_space for name in self.agents}
    
    @property
    def spec(self):
        return self.env.spec

    def reset(self):
        obs = self.env.reset()
        return [obs]

    def seed(self, seed):
        self.env.seed(seed)
        np.random.seed(seed)

    def render(self,):
        return self.env.render()

    def step(self, actions):
        assert len(actions) >= 1
        action = actions[0]
        noise = np.random.uniform(-1, 1, action.shape[0])
        action = action + 0. * noise
        obs, reward, done, info = self.env.step(action)
        obs = obs.squeeze() # for continuous gym envs it require squeeze()
        return [obs], [reward], [done], [info]

    def close(self):
        self.env.close()

class Gym2AgentAdversarialWrapper():
    """ Wrap single agent OpenAI gym game to be multi-agent (one adversarial) version """
    def __init__(self, env):
        super(Gym2AgentAdversarialWrapper, self).__init__()
        self.env = env
        self.agents = ['player', 'adversarial']
        self.num_agents = len(self.agents)
        self.observation_space = self.env.observation_space
        self.observation_spaces = {name: self.env.observation_space for name in self.agents}
        self.action_space = self.env.action_space
        self.action_spaces = {name: self.action_space for name in self.agents}
        self.adversarial_coef = 0.1 # adversarial action scale
        self.metadata = env.metadata
    
    @property
    def spec(self):
        return self.env.spec

    def reset(self):
        obs = self.env.reset()
        return [obs, obs]

    def seed(self, seed):
        self.env.seed(seed)
        np.random.seed(seed)

    def render(self,):
        return self.env.render()

    def step(self, actions):
        assert len(actions) >= 1
        actions = [a.squeeze() for a in actions]
        action = actions[0] + self.adversarial_coef * actions[1]  
        obs, reward, done, info = self.env.step(action)
        obs = obs.squeeze() # for continuous gym envs it require squeeze()
        return [obs, obs], [reward, -reward], [done, done], [info, info]

    def close(self):
        self.env.close()

# class SlimeVolleyWrapper(gym.Wrapper):
#     """ 
#     Wrapper to transform SlimeVolley environment (https://github.com/hardmaru/slimevolleygym) 
#     into PettingZoo (https://github.com/PettingZoo-Team/PettingZoo) env style. 
#     Specifically, most important changes are:
#     1. to make reset() return a dictionary of obsevations: {'agent1_name': obs1, 'agent2_name': obs2}
#     2. to make step() return dict of obs, dict of rewards, dict of dones, dict of infos, in a similar format as above.
#     """
#     # action transformation of SlimeVolley, the inner action is MultiBinary, which can be transformed into Discrete
#     action_table = [[0, 0, 0], # NOOP
#                     [1, 0, 0], # LEFT (forward)
#                     [1, 0, 1], # UPLEFT (forward jump)
#                     [0, 0, 1], # UP (jump)
#                     [0, 1, 1], # UPRIGHT (backward jump)
#                     [0, 1, 0]] # RIGHT (backward)


#     def __init__(self, env, against_baseline=False):
#         # super(SlimeVolleyWrapper, self).__init__()
#         super().__init__(env)
        # self.env = env
#         self.agents = ['first_0', 'second_0']
#         self.observation_space = self.env.observation_space
#         self.observation_spaces = {name: self.env.observation_space for name in self.agents}
#         self.action_space = gym.spaces.Discrete(len(self.action_table))
#         self.action_spaces = {name: self.action_space for name in self.agents}
#         self.against_baseline = against_baseline

    # @property
    # def spec(self):
    #     return self.env.spec

#     def reset(self, observation=None):
#         obs1 = self.env.reset()
#         obs2 = obs1 # both sides always see the same initial observation.
#         obs = {}
#         obs[self.agents[0]] = obs1
#         obs[self.agents[1]] = obs2
#         return obs

    # def seed(self, seed):
    #     self.env.seed(seed)
    #     np.random.seed(seed)

#     def render(self,):
#         self.env.render()

#     def step(self, actions):
#         obs, rewards, dones, infos = {},{},{},{}
#         actions_ = [self.env.discreteToBox(int(a)) for a in actions.values()]  # from discrete to multibinary action
#         if self.against_baseline:
#             obs1, reward, done, info = self.env.step(actions_[1])
#             obs0 = obs1
#             rewards[self.agents[0]] = -reward
#             rewards[self.agents[1]] = reward # the reward is for the learnable agent (second)
#         else:
#             # normal 2-player setting
#             if len(self.observation_space.shape)>1: 
#                 # for image-based env, fake the action list as one input to pass through NoopResetEnv, etc wrappers
#                 obs0, reward, done, info = self.env.step(actions_)
#             else:
#                 obs0, reward, done, info = self.env.step(*actions_)
#             obs1 = info['otherObs']
#             rewards[self.agents[0]] = reward
#             rewards[self.agents[1]] = -reward # the reward is for the learnable agent (second)
#         obs[self.agents[0]] = obs0
#         obs[self.agents[1]] = obs1
#         dones[self.agents[0]] = done
#         dones[self.agents[1]] = done
#         infos[self.agents[0]] = info
#         infos[self.agents[1]] = info

#         return obs, rewards, dones, infos

#     def close(self):
#         self.env.close()


# class SlimeVolleyWrapper(gym.Wrapper):
class SlimeVolleyWrapper():
    """ 
    Wrapper to transform SlimeVolley environment (https://github.com/hardmaru/slimevolleygym) 
    into PettingZoo (https://github.com/PettingZoo-Team/PettingZoo) env style. 
    Specifically, most important changes this wrapper makes are:

        1. to make reset() return a dictionary of obsevations: {'agent1_name': obs1, 'agent2_name': obs2}.
        
        2. to make step() return dict of obs, dict of rewards, dict of dones, dict of infos, in a similar format as above.
    """
    # action transformation of SlimeVolley, the inner action is MultiBinary, which can be transformed into Discrete
    action_table = [[0, 0, 0], # NOOP
                    [1, 0, 0], # LEFT (forward)
                    [1, 0, 1], # UPLEFT (forward jump)
                    [0, 0, 1], # UP (jump)
                    [0, 1, 1], # UPRIGHT (backward jump)
                    [0, 1, 0]] # RIGHT (backward)

    def __init__(self, env, against_baseline=False):
        # super().__init__(env)
        self.env = env
        self.agents = ['second_0'] if against_baseline else ['first_0', 'second_0'] # when against baseline the learnable agent is on the right side (second)
        self.num_agents = len(self.agents)
        self.observation_space = self.env.observation_space
        self.observation_spaces = {name: self.env.observation_space for name in self.agents}
        self.action_space = gym.spaces.Discrete(len(self.action_table))
        self.action_spaces = {name: self.action_space for name in self.agents}
        self.against_baseline = against_baseline
        self.metadata = env.metadata

    @property
    def spec(self):
        return self.env.spec

    def reset(self, observation=None):
        """Resets the environment to an initial state and returns an initial observation.

        :return: a dictionary of observations for each agent
        :rtype: dict
        """        
        obs0 = self.env.reset()
        if self.against_baseline: 
            return {self.agents[0]: obs0}
        else:
            obs1 = obs0 # both sides always see the same initial observation.
            obs = {}
            obs[self.agents[0]] = obs0
            obs[self.agents[1]] = obs1
            return obs

    def seed(self, seed):
        """Set the seed for environment and numpy.

        :param seed: seed value
        :type seed: int
        """        
        self.env.seed(seed)
        np.random.seed(seed)

    def render(self,):
        """ Render the scene.
        """        
        return self.env.render()

    def step(self, actions):
        obss, rewards, dones, infos = {},{},{},{}
        actions_ = [self.env.discreteToBox(int(a)) for a in actions.values()]  # from discrete to multibinary action
        if self.against_baseline:
            obs, reward, done, info = self.env.step(actions_[0]) # the reward is for the learnable agent (on the right side)
            obss[self.agents[0]] = obs
            rewards[self.agents[0]] = reward
            dones[self.agents[0]] = done
            infos[self.agents[0]] = info
        else:
            # normal 2-player setting
            if len(self.observation_space.shape)>1: 
                # for image-based env, fake the action list as one input to pass through NoopResetEnv, etc wrappers
                obs0, reward, done, info = self.env.step(actions_)
            else:
                obs0, reward, done, info = self.env.step(*actions_)  # gym!=0.18 will break this line
            obs1 = info['otherObs']
            rewards[self.agents[0]] = reward
            rewards[self.agents[1]] = -reward 
            obss[self.agents[0]] = obs0
            obss[self.agents[1]] = obs1
            dones[self.agents[0]] = done
            dones[self.agents[1]] = done
            infos[self.agents[0]] = info
            infos[self.agents[1]] = info

        return obss, rewards, dones, infos

    def close(self):
        self.env.close()

class Dict2TupleWrapper():
    """ Wrap the PettingZoo envs to have a similar style as LaserFrame in NFSP """
    def __init__(self, env, keep_info=False):
        super(Dict2TupleWrapper, self).__init__()
        self.env = env
        self.num_agents = env.num_agents
        self.keep_info = keep_info  # if True keep info as dict
        self.metadata = env.metadata

        if len(env.observation_space.shape) > 1: # image
            old_shape = env.observation_space.shape
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8)
            self.obs_type = 'rgb_image'
        else:
            self.observation_space = env.observation_space
            self.obs_type = 'ram'
        self.action_space = env.action_space
        # self.observation_spaces = env.observation_spaces
        # self.action_spaces = env.action_spaces
        try:   # both pettingzoo and slimevolley can work with this
            self.agents = env.agents
        except:
            self.agents = env.unwrapped.agents
    
    @property
    def unwrapped(self,):
        return self.env

    @property
    def spec(self):
        return self.env.spec

    def observation_swapaxis(self, observation):
        return (np.swapaxes(observation[0], 2, 0), np.swapaxes(observation[1], 2, 0))
    
    def reset(self):
        obs_dict = self.env.reset()
        if self.obs_type == 'ram':
            return tuple(obs_dict.values())
        else:
            return self.observation_swapaxis(tuple(obs_dict.values()))

    def step(self, actions): 
        actions = {agent_name: action for agent_name, action in zip(self.agents, actions)}
        obs, rewards, dones, infos = self.env.step(actions)
        if self.obs_type == 'ram':
            o = tuple(obs.values())
        else:
            o = self.observation_swapaxis(tuple(obs.values()))
        r = list(rewards.values())
        d = list(dones.values())
        if self.keep_info:  # a special case for VectorEnv
            info = infos
        else:
            info = list(infos.values())
        del obs,rewards, dones, infos
        # r = self._zerosum_filter(r)

        return o, r, d, info

    # def _zerosum_filter(self, r):
    #     ## zero-sum filter: 
    #     # added for making non-zero sum game to be zero-sum, e.g. tennis_v2
    #     if np.sum(r) != 0:
    #         nonzero_idx = np.nonzero(r)[0][0]
    #         r[1-nonzero_idx] = -r[nonzero_idx]
    #     return r

    def seed(self, seed):
        try:
            self.env.seed(seed)
        except:
            self.env.reset(seed=seed)

    def render(self, mode='rgb_array'):
        frame = self.env.render(mode)
        return frame

    def close(self):
        self.env.close()
