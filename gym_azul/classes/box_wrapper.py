# From https://github.com/vladfi1/gym-dolphin/blob/master/dolphin/box_wrapper.py

import gym
from gym.spaces import *
from gym_azul.classes.tile import Tile

import numpy as np

__all__ = ['BoxWrapper']

class FlattenConvertor:
  def __init__(self, space):
    assert(isinstance(space, Box))
    self.in_space = space
    self.out_space = Box(space.low.flatten(), space.high.flatten())
  
  def __call__(self, x):
    assert(self.in_space.contains(x))
    return x.flatten()

class OneHotConvertor:
  def __init__(self, space):
    assert(isinstance(space, Discrete))
    self.in_space = space
    self.out_space = Box(0, 1, [space.n])
  
  def __call__(self, x):
    assert(self.in_space.contains(x))
    a = np.zeros([self.in_space.n])
    a[x] = 1
    return a

class TileConvertor:
  def __init__(self, space):
    assert(isinstance(space, Discrete))
    self.in_space = space
    self.out_space = Box(0, 1, [space.n - 1])
  
  def __call__(self, x):
    assert(self.in_space.contains(x))
    a = np.zeros([self.in_space.n])
    if x is not None:
      a[x] = 1
    return a


class MultiBinConvertor:
  def __init__(self, space):
    assert(isinstance(space, MultiBinary))
    self.in_space = space
    self.out_space = Box(0, 1, [space.n], dtype=int)
  
  def __call__(self, x):
    assert(self.in_space.contains(x))
    assert(self.out_space.contains(x))
    return x

class LinearIntConvertor:
  def __init__(self, space):
    assert(isinstance(space, Discrete))
    self.in_space = space
    self.out_space = Box(0, space.n - 1, [1], dtype=int)
  
  def __call__(self, x):
    assert(self.in_space.contains(x))
    assert(self.out_space.contains(x))
    return x


class ConcatConvertor:
  def __init__(self, space):
    assert(isinstance(space, Tuple) or isinstance(space, Dict))
    
    self.in_space = space
    if isinstance(space, Tuple):
        self.convertors = list(map(convertor, space.spaces))
    else:
        # Hardcode the different encoding for the "type" space
        items = sorted(space.spaces.items(),
                       key=lambda x: x[0].value if isinstance(x, Tile) else x[0])
        self.convertors = [TileConvertor(s) if n == "type" else convertor(s)
                           for n, s in items]

    low = np.concatenate([c.out_space.low for c in self.convertors])
    high = np.concatenate([c.out_space.high for c in self.convertors])
    
    self.out_space = Box(low, high)
  
  def __call__(self, xs):
    #assert(self.in_space.contains(xs))
    return np.concatenate([c(x) for c, x in zip(self.convertors, xs)])

def convertor(space):
  if isinstance(space, Box):
    return FlattenConvertor(space)
  elif isinstance(space, Discrete):
    return LinearIntConvertor(space)
    # return OneHotConvertor(space)
  elif isinstance(space, Tuple) or isinstance(space, Dict):
    return ConcatConvertor(space)
  elif isinstance(space, MultiBinary):
    return MultiBinConvertor(space)
  else:
    raise ValueError("Unsupported space %s" % space)

class BoxWrapper(gym.Wrapper):
  "Turns any observation space into a box."
  def __init__(self, env):
    super(BoxWrapper, self).__init__(env)
    self.convertor = convertor(env.observation_space)
    self.observation_space = self.convertor.out_space 

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    obs = self.convertor(obs)
    return obs, reward, done, info
  
  def reset(self):
    obs = self.env.reset()
    return self.convertor(obs)
