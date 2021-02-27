from keras import models
from keras import layers
from collections import deque
import gym
from keras import optimizers
import numpy as np
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy

env = gym.make("CartPole-v0")

i = env.observation_space.shape

env.seed(3)


m = models.Sequential()
m.add(layers.Dense(64, activation="linear", input_shape=((1, ) + i)))
m.add(layers.Dense(128, activation="linear"))
m.add(layers.Dense(128, activation="linear"))
m.add(layers.Dense(env.action_space.n))
  



#def get_q(state):
  #agent().predict(np.array(state).reshape([1, state.shape[0]])/255)[0]

def train():
  policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr="eps", value_max=0, value_min=-2, value_test=-2, nb_steps= 5000)
  memory = SequentialMemory(limit=50000, window_length=1)
  dqn = DQNAgent(model=m, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=5000, target_model_update=1e-2, policy=policy, nable_dueling_network=True, dueling_type='avg')
  dqn.compile(optimizer=optimizers.Adam(lr=0.001))
  dqn.fit(env, nb_steps=2000, visualize=True, verbose=1)
  dqn.test(env, nb_episodes=10, visualize=True)
  
train()


  
  







  