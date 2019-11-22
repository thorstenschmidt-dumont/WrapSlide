import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from gym import error, spaces, utils
from gym.utils import seeding
from bitstring import BitStream
import random
import copy
import gym_wrapslide

from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy


ENV_NAME = 'wrapslide-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

#input_shape = (18, 3, 1)
input_shape = (16,)
num_classes = nb_actions

model = Sequential()
model.add(Flatten(input_shape=(1,) + input_shape))
# model.add(Conv2D(128, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(Flatten())
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(num_classes, activation='sigmoid'))
#model.add(Dropout(0.5))
#model.add(Dense(num_classes, activation='softmax'))
model.summary()
"""
# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())
"""
# SARSA does not require a memory.
policy = BoltzmannQPolicy()
sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
sarsa.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
sarsa.fit(env, nb_steps=100000, visualize=False, verbose=2)
 
# After training is done, we save the final weights.
sarsa.save_weights('sarsa_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
sarsa.test(env, nb_episodes=50, visualize=False)
