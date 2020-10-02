# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:55:49 2020

@author: thorstens
"""

import numpy as np
from WrapSlide import Wrapslide
from FFSNeuralNetwork import FFSNNetwork

# Define my functions
def Convert(lst, W):
    """Convert from a list data type to a dictionary data type"""
    res_dct = {lst[i]: W[i] for i in range(0, len(lst))} 
    return res_dct 

def dict2weights(W):
    """Convert from dictionary data type to a weight vector"""
    data = list(W.items())
    an_array = np.array(data)
    array = an_array.flatten()
    for i in range(int(len(array)/2)):
        array1 = array[i*2+1]
        array1 = array1.flatten()
        if i == 0:
            vector = array1
        else:
            vector = np.concatenate((vector, array1))    
    return vector


def weights2dict(V, n_input, n_hidden):
    """Convert from a weight vector data type to a dictionary data type"""
    architecture = np.concatenate(([n_input], n_hidden, [1]))
    W = []
    for i in range(len(architecture)-1):
        if i == 0:
            W1 = V[0:(architecture[i]*architecture[i+1])]
            W1 = W1.reshape((architecture[i], architecture[i+1]))
            count = (architecture[i]*architecture[i+1])
        else:
            W1 = V[count:(count+(architecture[i]*architecture[i+1]))]
            W1 = W1.reshape((architecture[i], architecture[i+1]))
            count = (count+(architecture[i]*architecture[i+1]))
        W.append(W1)
    # Driver code 
    lst = range(1, len(architecture)) 
    dictionary = Convert(lst, W)
    return dictionary

def bias2dict(V, n_hidden):
    """Convert from a weight vector data type to a dictionary data type"""
    architecture = np.concatenate((n_hidden, [1]))
    W = []
    for i in range(len(architecture)):
        if i == 0:
            W1 = V[0:(architecture[i])]
        else:
            W1 = V[architecture[i-1]:((architecture[i-1]+architecture[i]))]
        W.append(W1)
    # Driver code 
    lst = range(1, (len(architecture)+1)) 
    dictionary = Convert(lst, W)
    return dictionary


def GenerateSwarm(SwarmSize, n_input, n_output, n_hidden):
    for i in range(SwarmSize):
        # Initialise a new random neural network
        ffsn = FFSNNetwork(n_input, n_hidden)
        # Recover the weights and bias from the random network
        W = ffsn.W
        B = ffsn.B
        # Convert the dictionary values to vectors
        weights = dict2weights(W)
        bias = dict2weights(B)
        # Concatenate the weight vectors into a particle
        particle = np.concatenate((weights, bias))
        # Combine the particles into a swarm
        if i == 0:
            Swarm = particle
        else:
            Swarm = np.concatenate((Swarm,particle))
    # Reshape the swarm to the right output format
    Swarm = Swarm.reshape((SwarmSize,(len(weights)+len(bias))))
    return Swarm


def EvaluateObjectiveSwarm(Swarm, weights, bias, n_input, n_output, n_hidden):
    Evaluations = 1
    MaxSteps = 20
    ObjValues = np.zeros((len(Swarm),Evaluations))
    ObjValue = np.zeros(len(Swarm))
    np.random.seed(1)
    for j in range(Evaluations):    
        size = Game.size
        stateReset = Game.reset()
        for i in range(len(Swarm)):
            # Determine the weights and bias components of each particle
            NNweights = Swarm[i, 0:weights]
            NNbias = Swarm[i, weights:int(weights + bias)]
            # Convert the weights and bias to dictionary format
            Wupdated = weights2dict(NNweights, n_input, n_hidden)
            Bupdated = bias2dict(NNbias, n_hidden)
            # Update the values within the neural network
            ffsn.W = Wupdated
            ffsn.B = Bupdated
            # Perform the prediction on the train set
            done = False
            steps = 0
            state = stateReset
            values = np.zeros(4*(size-1))
            while done == False and steps < 20:
                for i in range(actions):
                    Game.state = state
                    step = Game.step(i)
                    done = step[2]
                    stateNew = step[0]
                    x_val = stateNew.reshape((1,size**2))
                    values[i] = ffsn.predict(x_val)
                action = np.argmax(values)
                Game.state = state
                step = Game.step(action)
                steps += 1
                done = step[2]
                state = step[0]
            ObjValues[i, j] = MaxSteps - steps
    for i in range(len(Swarm)):
        ObjValue[i] = np.mean(ObjValues[i,:])
    #print(ObjValues)
    return ObjValue

def EvaluateObjective(ffsn):
    size = Game.size
    actions = 4*(size-1)
    done = False
    steps = 0
    values = np.zeros(actions)
    Evaluations = 10
    Objective = np.zeros(Evaluations)
    for j in range(Evaluations):
        stateReset = Game.reset()
        while done == False and steps < 20:
            state = stateReset
            for i in range(actions):
                Game.state = state
                step = Game.step(i)
                done = step[2]
                stateNew = step[0]
                x_val = stateNew.reshape((1,size**2))
                values[i] = ffsn.predict(x_val)
            action = np.argmax(values)
            Game.state = state
            step = Game.step(action)
            steps += 1
            done = step[2]
            state = step[0]
        print(done)
        Objective[j] = steps
    return Objective


def PlayGame(ffsn, state_in):
    MaxSteps = 50
    steps = 0
    done = False
    values = np.zeros(4*(Game.size-1))
    state = state_in
    while done == False and steps < MaxSteps:
        for i in range(actions):
            Game.state = state
            step = Game.step(i)
            done = step[2]
            stateNew = step[0]
            x_val = stateNew.reshape((1,size**2))
            values[i] = ffsn.predict(x_val)
        action = np.argmax(values)
        Game.state = state
        step = Game.step(action)
        steps += 1
        done = step[2]
        state = step[0]
    return (MaxSteps - steps)


def Objective(Swarm, weights, bias, n_input, n_output, n_hidden):
    Evaluations = 10
    Results = np.zeros((len(Swarm), Evaluations))
    ObjValue = np.zeros(len(Swarm))
    for j in range(Evaluations):
        state = Game.reset()
        for i in range(len(Swarm)):
            # Determine the weights and bias components of each particle
            NNweights = Swarm[i, 0:weights]
            NNbias = Swarm[i, weights:int(weights + bias)]
            # Convert the weights and bias to dictionary format
            Wupdated = weights2dict(NNweights, n_input, n_hidden)
            Bupdated = bias2dict(NNbias, n_hidden)
            # Update the values within the neural network
            ffsn.W = Wupdated
            ffsn.B = Bupdated
            # Perform the prediction on the train set
            Results[i, j] = PlayGame(ffsn, state)
    for j in range(len(Swarm)):
        ObjValue[j] = np.mean(Results[j, :])
    return ObjValue


Results = []
Game = Wrapslide()
size = Game.size
colours = Game.colours
actions = 4*(size-1)
state = Game.reset()
done = False
steps = 0
n_input = size**2
n_hidden = [6]
n_output = 1
Game.level = 3
SwarmSize = 10
Swarm = GenerateSwarm(SwarmSize, n_input, n_output, n_hidden)

ffsn = FFSNNetwork(n_input, n_hidden)
# Read the weights
W = ffsn.W
B = ffsn.B

# Generate the weight and bias vectors
weights = len(dict2weights(W))
bias = len(dict2weights(B))

objective = Objective(Swarm, weights, bias, n_input, n_output, n_hidden)
