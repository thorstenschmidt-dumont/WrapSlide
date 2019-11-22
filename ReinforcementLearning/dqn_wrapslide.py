import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, Permute, Input
from keras.optimizers import Adam
from gym import error, spaces, utils
from gym.utils import seeding
from bitstring import BitStream
import random
import copy
import gym_wrapslide
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, GreedyQPolicy,EpsGreedyQPolicy
from rl.memory import SequentialMemory
import keras.backend as K

ENV_NAME = 'wrapslide-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

obv = env.reset()
print(obv)
print(env.observation_space.shape)

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


'''
# Next, we build a very simple model.
print(env.observation_space.shape)
model = Sequential()
model.add(input_shape=(4,4))# + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())
'''

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=500000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=0.001), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)

# Lets load previously saved weights
#dqn.load_weights('dqn_wrapslide-v0_weights_4x4.h5f')

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights_4x4.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=False)
