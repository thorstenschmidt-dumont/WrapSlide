import numpy as np
import gym
import math

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
from rl.policy import BoltzmannQPolicy, GreedyQPolicy, MaxBoltzmannQPolicy, EpsGreedyQPolicy, GreedyQTestPolicy
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
size = int(nb_actions/4+1)
colours = env.colours
Neurons = 100
Layers = 1

#input_shape = (18, 3, 1)
input_shape = (size**2,)
#input_shape = (4,4,1)
num_classes = nb_actions

model = Sequential()
model.add(Flatten(input_shape=(1,) + input_shape))
model.add(Dense(Neurons, activation='sigmoid'))
#model.add(Dense(Neurons, activation='sigmoid'))
#model.add(Dense(Neurons, activation='sigmoid'))
model.add(Dense(num_classes, activation='sigmoid'))
model.summary()

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=500000, window_length=1)
policy = EpsGreedyQPolicy()
test_policy = GreedyQTestPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy, test_policy=test_policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

#dqn.load_weights('dqn_{}_weights_3Col.h5f'.format(ENV_NAME))

#Now lets learn something
#dqn.fit(env, nb_steps=1000, visualize=False, verbose=2)

# After training is done, we save the final weights.
#dqn.save_weights('dqn_{}_weights_{}Col_{}Neurons_{}Layers_{}x.h5f'.format(ENV_NAME,colours,Neurons,Layers,size), overwrite=True)
#dqn.save_weights('dqn_{}_weights_3Col.h5f'.format(ENV_NAME), overwrite=True)

for i in range(50):
    print(i)
    dqn.load_weights('dqn_{}_weights_3Col.h5f'.format(ENV_NAME))
    
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=1000, visualize=False, verbose=2)
    
    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights_3Col.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
#dqn.test(env, nb_episodes=100, visualize=False)
