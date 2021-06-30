import gym
import numpy as np
from collections import deque

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

