from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym
import gym_wrapslide
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy, GreedyQTestPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='wrapslide-v0')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
ENV_NAME = 'wrapslide-v0'
size = int(nb_actions/4+1)
colours = env.colours
Neurons = 100
Layers = 3
Conv = 1

INPUT_SHAPE = (size, size)
WINDOW_LENGTH = 1

class AtariProcessor(Processor):
    def process_observation(self, observation):
        #print("This is the observation", observation)
        assert observation.ndim == 2  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE)  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 1.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()
model.add(Permute((2, 3, 1), input_shape=input_shape))
model.add(Convolution2D(size**2, (int(size/2), int(size/2)), strides=(2, 2)))
model.add(Activation('relu'))
#model.add(Convolution2D(16, (1, 1), strides=(1, 1)))
#model.add(Activation('relu'))
#model.add(Convolution2D(64, (2, 2), strides=(2, 1)))
#model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('sigmoid'))
model.add(Dense(100))
model.add(Activation('sigmoid'))
model.add(Dense(100))
model.add(Activation('sigmoid'))
model.add(Dense(nb_actions))
model.add(Activation('sigmoid'))
model.build()
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=500000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
#policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
#                              nb_steps=1000000)
policy = EpsGreedyQPolicy()
test_policy = GreedyQTestPolicy()
# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, test_policy = test_policy, memory=memory,
               processor=processor, nb_steps_warmup=10, gamma=.99, target_model_update=1e-2)#,
               #train_interval=1, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

#dqn.load_weights('dqn_{}_weights_{}Col_{}Neurons_{}Layers_{}x_Convnet_Sigmoid.h5f'.format(ENV_NAME,colours,Neurons,Layers,size))
  
#Now lets learn something
dqn.fit(env, nb_steps=1000, visualize=False, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights_{}Col_{}Conv_{}Neurons_{}Layers_{}x_Convnet_Sigmoid.h5f'.format(ENV_NAME,colours,Conv,Neurons,Layers,size), overwrite=True)

for j in range(6):
    env.level = j + 1
    for i in range(50):
        print(i)
        dqn.load_weights('dqn_{}_weights_{}Col_{}Conv_{}Neurons_{}Layers_{}x_Convnet_Sigmoid.h5f'.format(ENV_NAME,colours,Conv,Neurons,Layers,size))
        
        # Okay, now it's time to learn something! We visualize the training here for show, but this
        # slows down training quite a lot. You can always safely abort the training prematurely using
        # Ctrl + C.
        dqn.fit(env, nb_steps=1000, visualize=False, verbose=2)
        
        # After training is done, we save the final weights.
        dqn.save_weights('dqn_{}_weights_{}Col_{}Conv_{}Neurons_{}Layers_{}x_Convnet_Sigmoid.h5f'.format(ENV_NAME,colours,Conv,Neurons,Layers,size), overwrite=True)

env.initialise = False
dqn.load_weights('dqn_{}_weights_{}Col_{}Conv_{}Neurons_{}Layers_{}x_Convnet_Sigmoid.h5f'.format(ENV_NAME,colours,Conv,Neurons,Layers,size))
dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2)
dqn.save_weights('dqn_{}_weights_{}Col_{}Conv_{}Neurons_{}Layers_{}x_Convnet_Sigmoid.h5f'.format(ENV_NAME,colours,Conv,Neurons,Layers,size), overwrite=True)
dqn.load_weights('dqn_{}_weights_{}Col_{}Conv_{}Neurons_{}Layers_{}x_Convnet_Sigmoid.h5f'.format(ENV_NAME,colours,Conv,Neurons,Layers,size))

# Finally, evaluate our algorithm for 5 episodes.
env.test = True
dqn.test(env, nb_episodes=100, visualize=False)