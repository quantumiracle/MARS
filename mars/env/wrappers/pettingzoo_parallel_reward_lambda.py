##
# Pettingzoo's reward_lambda_v0 only supports individually change of the rewards,
# we want a simultaneous change of all agent's reward, so providing this function.
# ref: https://github.com/Farama-Foundation/SuperSuit/blob/master/supersuit/lambda_wrappers/reward_lambda.py
##

from supersuit.utils.base_aec_wrapper import PettingzooWrap
from supersuit.utils.wrapper_chooser import WrapperChooser
import gym
from supersuit.utils.make_defaultdict import make_defaultdict
import numpy as np

class aec_reward_lambda(PettingzooWrap):
    def __init__(self, env, change_reward_fn):
        assert callable(
            change_reward_fn
        ), "change_reward_fn needs to be a function. It is {}".format(change_reward_fn)
        self._change_reward_fn = change_reward_fn
        self.env = env
        super().__init__(env)

    def _check_wrapper_params(self):
        pass

    def _modify_spaces(self):
        pass

    def reset(self, seed=None, return_info=False, options=None):
        # super().reset(seed=seed, options=options)
        super().reset(seed=seed)
        changed_rewards = self._change_reward_fn(list(self.rewards.values()))
        self.rewards = {
            agent: reward
            for agent, reward in zip(list(self.rewards.keys()), changed_rewards)
        }
        self.__cumulative_rewards = make_defaultdict({a: 0 for a in self.agents})
        self._accumulate_rewards()

    def step(self, action):
        agent = self.env.agent_selection
        super().step(action)
        changed_rewards = self._change_reward_fn(list(self.rewards.values()))
        self.rewards = {
            agent: reward
            for agent, reward in zip(list(self.rewards.keys()), changed_rewards)
        }
        self.__cumulative_rewards[agent] = 0
        self._cumulative_rewards = self.__cumulative_rewards
        self._accumulate_rewards()

    def render(self, mode):
        return self.env.render(mode)



class gym_reward_lambda(gym.Wrapper):
    def __init__(self, env, change_reward_fn):
        assert callable(
            change_reward_fn
        ), "change_reward_fn needs to be a function. It is {}".format(change_reward_fn)
        self._change_reward_fn = change_reward_fn
        self.env = env
        super().__init__(env)

    def step(self, action):
        obs, rew, done, info = super().step(action)
        return obs, self._change_reward_fn(rew), done, info

    def render(self, mode):
        return self.env.render(mode)


reward_lambda_v1 = WrapperChooser(
    aec_wrapper=aec_reward_lambda, gym_wrapper=gym_reward_lambda
)