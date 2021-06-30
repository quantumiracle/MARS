import random
import gym
import numpy as np

class PettingzooClassicWrapper():
    def __init__(self, env, observation_mask=1.):  
        """
        Args:

            observation_mask: mask the observation to be anyvalue, if None, no mask. 
        """
        super(PettingzooClassicWrapper, self).__init__()
        self.env = env
        self.observation_mask = observation_mask
        self.action_spaces = self.env.action_spaces
        self.action_space = list(self.action_spaces.values())[0]
        self.agents = list(self.action_spaces.keys())

        # for rps_v1, discrete to box, fake space
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.uint8)
        self.observation_spaces = {a:self.observation_space for a in self.agents}

        # for holdem
        # obs_space = self.env.observation_spaces.values()
        # obs_len = obs_space.shape[0]-action_space.n
        # self.observation_spaces = Box(shape=(obs_len,),low=obs_space.low[:obs_len],high=obs_space.high[:obs_len])

    def reset(self, observation=None):
        obs = self.env.reset()
        if self.observation_mask is not None:
            return {a: self.observation_mask for a in self.agents}
        else:
            return obs

    def seed(self, SEED):
        self.env.seed(SEED)

    def render(self,):
        self.env.render()

    def close(self):
        self.env.close()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self.observation_mask is not None:  # masked observation with a certain value
            observation = {a:self.observation_mask for a in observation.keys()}
        return observation, reward, done, info


class PettingzooClassic_Iterate2Parallel():
    def __init__(self, env, observation_mask=1.):  
        """
        Args:

            observation_mask: mask the observation to be anyvalue, if None, no mask. 
        """
        super(PettingzooClassic_Iterate2Parallel, self).__init__()
        self.env = env
        self.observation_mask = observation_mask
        self.action_spaces = self.env.action_spaces
        self.action_space = list(self.action_spaces.values())[0]
        self.agents = list(self.action_spaces.keys())

        self.observation_space = list(self.env.observation_spaces.values())[0]['observation']
        self.observation_spaces = {a:self.observation_space for a in self.agents}

    def reset(self, observation=None):
        obs = self.env.reset()
        if self.observation_mask is not None:
            return {a: self.observation_mask for a in self.agents}
        else:
            if obs is None:
                return {a:np.zeros(self.observation_space.shape[0]) for a in self.agents} # return zeros
            else:
                return {a: obs[a]['observation'] for a in self.agents}

    def seed(self, SEED):
        self.env.seed(SEED)

    def render(self,):
        self.env.render()

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

class Atari2AgentWrapper():
    """ Wrap single agent OpenAI gym atari game to be two-agent version """
    def __init__(self, env, keep_info=False):
        super(Atari2AgentWrapper, self).__init__()
        self.env = env
        self.keep_info = keep_info
        self.agents = ['first_0', 'second_0']
        self.observation_space = self.env.observation_space
        self.observation_spaces = {name: self.env.observation_space for name in self.agents}
        self.action_space = self.env.action_space
        self.action_spaces = {name: self.action_space for name in self.agents}

    def reset(self, observation=None):
        obs1 = self.env.reset()
        return (obs1, obs1)

    def seed(self, SEED):
        self.env.seed(SEED)

    def render(self,):
        self.env.render()

    def step(self, actions):
        action = list(actions.values())[0]
        next_state, reward, done, info = self.env.step(action)
        if self.keep_info:
            return [next_state, next_state], [reward, reward], done, info
        else:
            return [next_state, next_state], [reward, reward], done, [info, info]

    def close(self):
        self.env.close()

class SlimeVolleyWrapper(gym.Wrapper):
    """ 
    Wrapper to transform SlimeVolley environment (https://github.com/hardmaru/slimevolleygym) 
    into PettingZoo (https://github.com/PettingZoo-Team/PettingZoo) env style. 
    Specifically, most important changes are:
    1. to make reset() return a dictionary of obsevations: {'agent1_name': obs1, 'agent2_name': obs2}
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
        # super(SlimeVolleyWrapper, self).__init__()
        super().__init__(env)
        self.env = env
        self.agents = ['first_0', 'second_0']
        self.observation_space = self.env.observation_space
        self.observation_spaces = {name: self.env.observation_space for name in self.agents}
        self.action_space = gym.spaces.Discrete(len(self.action_table))
        self.action_spaces = {name: self.action_space for name in self.agents}
        self.against_baseline = against_baseline

    def reset(self, observation=None):
        obs1 = self.env.reset()
        obs2 = obs1 # both sides always see the same initial observation.
        obs = {}
        obs[self.agents[0]] = obs1
        obs[self.agents[1]] = obs2
        return obs

    def seed(self, SEED):
        self.env.seed(SEED)

    def render(self,):
        self.env.render()

    def step(self, actions):
        obs, rewards, dones, infos = {},{},{},{}
        actions_ = [self.env.discreteToBox(a) for a in actions.values()]  # from discrete to multibinary action
        if self.against_baseline:
            # this is for validation: load a single policy as 'second_0' to play against the baseline agent (via self-play in 2015)
            obs2, reward, done, info = self.env.step(actions_[1]) # extra argument
            obs1 = obs2 
        else:
            # normal 2-player setting
            if len(self.observation_space.shape)>1: 
                # for image-based env, fake the action list as one input to pass through NoopResetEnv, etc wrappers
                obs1, reward, done, info = self.env.step(actions_) # extra argument
            else:
                obs1, reward, done, info = self.env.step(*actions_) # extra argument
            obs2 = info['otherObs']

        obs[self.agents[0]] = obs1
        obs[self.agents[1]] = obs2
        rewards[self.agents[0]] = -reward
        rewards[self.agents[1]] = reward # the reward is for the learnable agent (second)
        dones[self.agents[0]] = done
        dones[self.agents[1]] = done
        infos[self.agents[0]] = info
        infos[self.agents[1]] = info

        return obs, rewards, dones, infos

    def close(self):
        self.env.close()

class Dict2TupleWrapper():
    """ Wrap the PettingZoo envs to have a similar style as LaserFrame in NFSP """
    def __init__(self, env, keep_info=False):
        super(Dict2TupleWrapper, self).__init__()
        self.env = env
        self.keep_info = keep_info  # if True keep info as dict
        if len(env.observation_space.shape) > 1: # image
            old_shape = env.observation_space.shape
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8)
            self.obs_type = 'rgb_image'
        else:
            self.observation_space = env.observation_space
            self.obs_type = 'ram'
        self.action_space = env.action_space
        fake_env = gym.make('Pong-v0')
        self.spec = fake_env.spec
        try:   # both pettingzoo and slimevolley can work with this
            self.agents = env.agents
        except:
            self.agents = env.unwrapped.agents
        try:
            self.spec.id = env.env.spec.id
        except:
            pass
        fake_env.close()

    def observation_swapaxis(self, observation):
        return (np.swapaxes(observation[0], 2, 0), np.swapaxes(observation[1], 2, 0))
    
    def reset(self):
        obs_dict = self.env.reset()
        if self.obs_type == 'ram':
            return tuple(obs_dict.values())
        else:
            return self.observation_swapaxis(tuple(obs_dict.values()))

    def step(self, actions): 
        obs, rewards, dones, infos = self.env.step(actions)
        if self.obs_type == 'ram':
            o = tuple(obs.values())
        else:
            o = self.observation_swapaxis(tuple(obs.values()))
        r = list(rewards.values())
        d = np.any(np.array(list(dones.values())))  # if done is True for any player, it is done for the game
        if self.keep_info:  # a special case for VectorEnv
            info = infos
        else:
            info = list(infos.values())
        del obs,rewards, dones, infos
        return o, r, d, info

    def seed(self, SEED):
        self.env.seed(SEED)

    def render(self,):
        self.env.render()

    def close(self):
        self.env.close()
