import gym
import numpy as np
from collections import deque
import cv2
from typing import Any, Callable
class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to num_channels x weight x height
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return (np.swapaxes(observation[0], 2, 0), np.swapaxes(observation[1], 2, 0))
    

def wrap_pytorch(env):
    return ImageToPyTorch(env)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FrameStack(gym.Wrapper):
  def __init__(self, env, n_frames):
    """
    Stack n_frames last frames.     (don't use lazy frames)
    or alternatively, run:
    from slimevolleygym import FrameStack # doesn't use Lazy Frames, easier to debug

    modified from:
    stable_baselines.common.atari_wrappers
    :param env: (Gym Environment) the environment
    :param n_frames: (int) the number of frames to stack
    """
    gym.Wrapper.__init__(self, env)
    self.n_frames = n_frames
    self.frames = deque([], maxlen=n_frames)
    shp = env.observation_space.shape
    self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * n_frames),
                                        dtype=env.observation_space.dtype)

  def reset(self):
    obs = self.env.reset()
    for _ in range(self.n_frames):
        self.frames.append(obs)
    return self._get_ob()

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self.frames.append(obs)
    return self._get_ob(), reward, done, info

  def _get_ob(self):
    assert len(self.frames) == self.n_frames
    return np.concatenate(list(self.frames), axis=2)

class NoopResetEnv(gym.Wrapper):
  def __init__(self, env, noop_max=30):
    """
    (from stable-baselines)
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    :param env: (Gym Environment) the environment to wrap
    :param noop_max: (int) the maximum value of no-ops to run
    """
    gym.Wrapper.__init__(self, env)
    self.noop_max = noop_max
    self.override_num_noops = None
    self.noop_action = 0
    assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

  def reset(self, **kwargs):
    self.env.reset(**kwargs)
    if self.override_num_noops is not None:
      noops = self.override_num_noops
    else:
      noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
    assert noops > 0
    obs = None
    for _ in range(noops):
      obs, _, done, _ = self.env.step(self.noop_action)
      if done:
        obs = self.env.reset(**kwargs)
    return obs

  def step(self, action):
    try: # modified
        output = self.env.step(action)
    except:
        output = self.env.step(*action) # expand the action if it is a list of two
    return output

class MaxAndSkipEnv(gym.Wrapper):
  def __init__(self, env, skip=4):
    """
    (from stable baselines)
    Return only every `skip`-th frame (frameskipping)
    :param env: (Gym Environment) the environment
    :param skip: (int) number of `skip`-th frame
    """
    gym.Wrapper.__init__(self, env)
    # most recent raw observations (for max pooling across time steps)
    self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=env.observation_space.dtype)
    self._skip = skip

  def step(self, action):
    """
    Step the environment with the given action
    Repeat action, sum reward, and max over last observations.
    :param action: ([int] or [float]) the action
    :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
    """
    total_reward = 0.0
    done = None
    for i in range(self._skip):
      obs, reward, done, info = self.env.step(action)
      if i == self._skip - 2:
        self._obs_buffer[0] = obs
      if i == self._skip - 1:
        self._obs_buffer[1] = obs
      total_reward += reward
      if done:
        break
    # Note that the observation on the done=True frame
    # doesn't matter
    max_frame = self._obs_buffer.max(axis=0)
    return max_frame, total_reward, done, info

  def reset(self, **kwargs):
      return self.env.reset(**kwargs)

class WarpFrame(gym.ObservationWrapper):
  def __init__(self, env):
    """
    (from stable-baselines)
    Warp frames to 84x84 as done in the Nature paper and later work.
    :param env: (Gym Environment) the environment
    """
    gym.ObservationWrapper.__init__(self, env)
    self.width = 84
    self.height = 84
    self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 1),
                                        dtype=env.observation_space.dtype)

  def observation(self, frame):
    """
    returns the current observation from a frame
    :param frame: ([int] or [float]) environment frame
    :return: ([int] or [float]) the observation
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
    return frame[:, :, None]

# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class NormalizeObservation(gym.core.Wrapper):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.
    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.
        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        # obs, rews, terminateds, truncateds, infos = self.env.step(action)
        obs, rews, terminateds, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        # return obs, rews, terminateds, truncateds, infos
        return obs, rews, terminateds, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)

        if self.is_vector_env:
            return self.normalize(obs), info
        else:
            return self.normalize(np.array([obs]))[0], info

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)



class NormalizeReward(gym.core.Wrapper):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.
    The exponential moving average will have variance :math:`(1 - \gamma)^2`.
    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.
        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        # obs, rews, terminateds, truncateds, infos = self.env.step(action)
        obs, rews, terminateds, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = np.array([rews])
        self.returns = self.returns * self.gamma + rews
        rews = self.normalize(rews)
        # dones = np.logical_or(terminateds, truncateds)
        dones = terminateds
        self.returns[dones] = 0.0
        if not self.is_vector_env:
            rews = rews[0]
        # return obs, rews, terminateds, truncateds, infos
        return obs, rews, terminateds, infos

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)



class TransformObservation(gym.ObservationWrapper):
    """Transform the observation via an arbitrary function :attr:`f`.
    The function :attr:`f` should be defined on the observation space of the base environment, ``env``, and should, ideally, return values in the same space.
    If the transformation you wish to apply to observations returns values in a *different* space, you should subclass :class:`ObservationWrapper`, implement the transformation, and set the new observation space accordingly. If you were to use this wrapper instead, the observation space would be set incorrectly.
    Example:
        >>> import gym
        >>> import numpy as np
        >>> env = gym.make('CartPole-v1')
        >>> env = TransformObservation(env, lambda obs: obs + 0.1*np.random.randn(*obs.shape))
        >>> env.reset()
        array([-0.08319338,  0.04635121, -0.07394746,  0.20877492])
    """

    def __init__(self, env: gym.Env, f: Callable[[Any], Any]):
        """Initialize the :class:`TransformObservation` wrapper with an environment and a transform function :param:`f`.
        Args:
            env: The environment to apply the wrapper
            f: A function that transforms the observation
        """
        super().__init__(env)
        assert callable(f)
        self.f = f

    def observation(self, observation):
        """Transforms the observations with callable :attr:`f`.
        Args:
            observation: The observation to transform
        Returns:
            The transformed observation
        """
        return self.f(observation)