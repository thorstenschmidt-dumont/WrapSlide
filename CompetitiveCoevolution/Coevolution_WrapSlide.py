# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 11:57:15 2020

@author: thorstens
"""

from MultiClassNetwork import FFSN_MultiClass
from WrapSlide import Wrapslide
import numpy as np
import random

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


def weights2dict(V, n_input, n_output, n_hidden):
    """Convert from a weight vector data type to a dictionary data type"""
    architecture = np.concatenate(([n_input], n_hidden, [n_output]))
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

def bias2dict(V, n_output, n_hidden):
    """Convert from a weight vector data type to a dictionary data type"""
    architecture = np.concatenate((n_hidden, [n_output]))
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
        ffsn_multi = FFSN_MultiClass(n_input, n_output, n_hidden)
        # Recover the weights and bias from the random network
        W = ffsn_multi.W
        B = ffsn_multi.B
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


def EvaluateObjective(Swarm, weights, bias, n_input, n_output, n_hidden):
    Evaluations = 10
    MaxSteps = 20
    ObjValues = np.zeros((len(Swarm),Evaluations))
    ObjValue = np.zeros(len(Swarm))
    random.seed(1)
    for j in range(Evaluations):    
        size = Game.size
        stateReset = Game.reset()
        for i in range(len(Swarm)):
            # Determine the weights and bias components of each particle
            NNweights = Swarm[i, 0:weights]
            NNbias = Swarm[i, weights:int(weights + bias)]
            # Convert the weights and bias to dictionary format
            Wupdated = weights2dict(NNweights, n_input, n_output, n_hidden)
            Bupdated = bias2dict(NNbias, n_output, n_hidden)
            # Update the values within the neural network
            ffsn_multi.W = Wupdated
            ffsn_multi.B = Bupdated
            # Perform the prediction on the train set
            done = False
            steps = 0
            state = stateReset
            while done == False and steps < MaxSteps:
                x_val = state.reshape((1,size**2))
                action = ffsn_multi.predict(x_val)
                action = np.argmax(action, 0)
                step = Game.step(action)
                steps += 1
                done = step[2]
                state = step[0]
            ObjValues[i, j] = MaxSteps - steps
    for i in range(len(Swarm)):
        ObjValue[i] = np.mean(ObjValues[i,:])
    #print(ObjValues)
    return ObjValue


def PlayGame(ffsn, state_in):
    MaxSteps = 20
    steps = 0
    done = False
    state = state_in
    statelist = []
    stateM = state.reshape(size,size)
    canonical = Game.findcanonical(stateM)
    statelist.append(canonical)
    while done == False and steps < MaxSteps:
        x_val = state.reshape((1,size**2))
        action = ffsn.predict(x_val)
        action1 = np.argmax(action, 0)
        step = Game.step(action1)
        k = 0
        canonical = Game.findcanonical(step[0].reshape(size,size))
        while canonical in statelist and k < 4*(size-1):
            action1 = action.argsort()[-k]
            Game.state = state
            step = Game.step(action1)
            canonical = Game.findcanonical(step[0].reshape(size,size))
            k += 1
        if k == 4*(size-1):
            step = Game.step(np.argmax(action))
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
            Wupdated = weights2dict(NNweights, n_input, n_output, n_hidden)
            Bupdated = bias2dict(NNbias, n_output, n_hidden)
            # Update the values within the neural network
            ffsn_multi.W = Wupdated
            ffsn_multi.B = Bupdated
            # Perform the prediction on the train set
            Results[i, j] = PlayGame(ffsn_multi, state)
    for j in range(len(Swarm)):
        ObjValue[j] = np.mean(Results[j, :])
    return ObjValue


def PSO(n_input, n_output, n_hidden, first, Swarm, SwarmSize):
    """Execute the Particle Swarm Optimisation algorithm"""
    # Initialisation
    w = 0.7     # inertia weight
    c1 = 1.4
    c2 = 1.4
    rho = 1
    architecture = np.concatenate(([n_input], n_hidden, [n_output]))
    size = 0
    weights = 0
    bias = 0
    normal = False
    normalised = False
    component = True
    slope = 0.025
    for i in range(len(architecture)-1):
        weights = weights + architecture[i]*architecture[i+1]
        bias = bias + architecture[i+1]
    size = weights + bias
    Velocity = np.zeros((SwarmSize, size))
    MaxIterations = 500
    if first == True:
        Swarm = GenerateSwarm(SwarmSize, n_input, n_output, n_hidden)
    GBest = np.zeros(size+1)
    PBest = np.zeros((SwarmSize, size+1))
    ObjValue = Objective(Swarm, weights, bias, n_input, n_output, n_hidden)
    #print(ObjValue)
    PBest = np.column_stack([Swarm, ObjValue])
    SwarmValue = np.column_stack([Swarm, ObjValue])
    
    for i in range(size):
        GBest[i] = Swarm[0, i]

    MeanVelocity = []
    iterations = 0
    Tracker = 0
    while iterations <= MaxIterations and Tracker <= 50:
        iterations += 1
        Tracker += 1
        Vmax = 0.75#1/(1 + np.exp(-slope*(iterations - (MaxIterations/2))))
        print("Search iterations: ", iterations)
        # Determine GBest
        for i in range(SwarmSize):
            if PBest[i, size] > GBest[size]:
                for j in range(size+1):
                    GBest[j] = PBest[i, j]
        
        for i in range(SwarmSize):
            if normal == True:
                # No velocity clamping
                for j in range(size):
                    r1 = np.random.random()
                    r2 = np.random.random()
                    if SwarmValue[i, size] == GBest[size]:
                        Velocity[i, j] = rho*(np.random.random()*0.01-0.005)
                    else:
                        Velocity[i, j] = w*Velocity[i, j] + c1*r1*(PBest[i, j]-SwarmValue[i, j]) + c2*r2*(GBest[j]-SwarmValue[i, j])
            if normalised == True:
                # Nomalised velocity clamping
                for j in range(size):
                    r1 = np.random.random()
                    r2 = np.random.random()
                    if SwarmValue[i, size] == GBest[size]:
                        Velocity[i, j] = rho*(np.random.random()*0.01-0.005)
                    else:
                        Velocity[i, j] = w*Velocity[i, j] + c1*r1*(PBest[i, j]-SwarmValue[i, j]) + c2*r2*(GBest[j]-SwarmValue[i, j])
                if np.linalg.norm(Velocity[i, :]) > Vmax:
                    Velocity[i, :] = (Vmax/np.linalg.norm(Velocity[i, :]))*Velocity[i, :]    
            if component == True:
                 # Componenet-wise velocity clamping
                for j in range(size):
                    r1 = np.random.random()
                    r2 = np.random.random()
                    if SwarmValue[i, size] == GBest[size]:
                        Velocity[i, j] = rho*(np.random.random()*0.01-0.005)
                    elif w*Velocity[i, j] + c1*r1*(PBest[i, j]-SwarmValue[i, j]) + c2*r2*(GBest[j]-SwarmValue[i, j]) > Vmax:
                        Velocity[i, j] = Vmax
                    elif w*Velocity[i, j] + c1*r1*(PBest[i, j]-SwarmValue[i, j]) + c2*r2*(GBest[j]-SwarmValue[i, j]) < -Vmax:
                        Velocity[i, j] = -Vmax
                    else:
                        Velocity[i, j] = w*Velocity[i, j] + c1*r1*(PBest[i, j]-SwarmValue[i, j]) + c2*r2*(GBest[j]-SwarmValue[i, j])
         
        # Move particles
        for i in range(SwarmSize):
            for j in range(size):
                Swarm[i, j] = Velocity[i, j] + Swarm[i, j]

        # Determine new objective value
        ObjValue = Objective(Swarm, weights, bias, n_input, n_output, n_hidden)
        print(ObjValue)
        SwarmValue = np.column_stack([Swarm, ObjValue])
        # Determine PBest
        for i in range(SwarmSize):
            if PBest[i, size] < SwarmValue[i, size]:
                PBest[i, :] = SwarmValue[i, :]

        # Determine GBest
        for i in range(SwarmSize):
            if PBest[i, size] > GBest[size]:
                for j in range(size+1):
                    GBest[j] = PBest[i, j]
                    Tracker = 0
        MeanVelocity.append(np.mean(np.absolute(Velocity)))
    MeanVelocity = np.array(MeanVelocity)
    # MeanVel = np.mean(MeanVelocity)
    return GBest, PBest

##############################################################################

# hidden sizes = [layer 1 size, layer 2 size, etc]
Game = Wrapslide()
size = Game.size
colours = Game.colours
actions = 4*(size-1)
state = Game.reset()
done = False
steps = 0
n_input = size**2
n_output = actions
n_hidden = [20]
SwarmSize = 30
first = False
Game.level = 1

# Generate the NN and its associated structure
ffsn_multi = FFSN_MultiClass(n_input, n_output, n_hidden)

# Read the weights
W = ffsn_multi.W
B = ffsn_multi.B

# Generate the weight and bias vectors
weights = dict2weights(W)
bias = dict2weights(B)

# Combine weights and bias into a single particle
particle = np.concatenate((weights, bias))

# Break up the particle into weight and bias components
weights = particle[0:len(weights)]
bias = particle[len(weights):len(particle)]

first = True

Swarm = np.zeros((SwarmSize,(len(weights)+len(bias))))
OptimalNetwork, PBest = PSO(n_input, n_output, n_hidden, first, Swarm, SwarmSize)
Swarm = PBest[:,0:(len(weights)+len(bias))]
print(Swarm)
    
# Break up the GBest particle into weight and bias components
weights = OptimalNetwork[0:len(weights)]
bias = OptimalNetwork[len(weights):len(particle)]
    
Wupdated = weights2dict(weights, n_input, n_output, n_hidden)
Bupdated = bias2dict(bias, n_output, n_hidden)
    
ffsn_multi.W = Wupdated
ffsn_multi.B = Bupdated

# Let's play
print("Let's play")
Games = np.zeros(100)
for j in range(100):
    state = Game.reset()
    done = False
    statelist = []
    stateM = state.reshape(size,size)
    canonical = Game.findcanonical(stateM)
    statelist.append(canonical)
    while done == False and steps < 100:
        x_val = state.reshape((1,size**2))
        action = ffsn_multi.predict(x_val)
        action1 = np.argmax(action, 0)
        step = Game.step(action1)
        k = 0
        canonical = Game.findcanonical(step[0].reshape(size,size))
        while canonical in statelist and k < 4*(size-1):
            action1 = action.argsort()[-k]
            Game.state = state
            step = Game.step(action1)
            canonical = Game.findcanonical(step[0].reshape(size,size))
            k += 1
        if k == 4*(size-1):
            step = Game.step(np.argmax(action))
        steps += 1
        done = step[2]
        state = step[0]
    Games[j] = steps
    print("Iteration ",j," solved in ",steps)

for i in range(5):
    first = False
    Game.level = i + 2
    # Let the optimisation begin
    #Swarm = np.zeros((SwarmSize,(len(weights)+len(bias))))
    OptimalNetwork, PBest = PSO(n_input, n_output, n_hidden, first, Swarm, SwarmSize)
    Swarm = PBest[:,0:(len(weights)+len(bias))]
    print(Swarm)
    
    # Break up the GBest particle into weight and bias components
    weights = OptimalNetwork[0:len(weights)]
    bias = OptimalNetwork[len(weights):len(particle)]
    
    Wupdated = weights2dict(weights, n_input, n_output, n_hidden)
    Bupdated = bias2dict(bias, n_output, n_hidden)
    
    ffsn_multi.W = Wupdated
    ffsn_multi.B = Bupdated
    
    # Let's play
    print("Let's play")
    Games = np.zeros(100)
    for j in range(100):
        state = Game.reset()
        done = False
        statelist = []
        stateM = state.reshape(size,size)
        canonical = Game.findcanonical(stateM)
        statelist.append(canonical)
        while done == False and steps < 100:
            x_val = state.reshape((1,size**2))
            action = ffsn_multi.predict(x_val)
            action1 = np.argmax(action, 0)
            step = Game.step(action1)
            k = 0
            canonical = Game.findcanonical(step[0].reshape(size,size))
            while canonical in statelist and k < 4*(size-1):
                action1 = action.argsort()[-k]
                Game.state = state
                step = Game.step(action1)
                canonical = Game.findcanonical(step[0].reshape(size,size))
                k += 1
            if k == 4*(size-1):
                step = Game.step(np.argmax(action))
            steps += 1
            done = step[2]
            state = step[0]
        Games[j] = steps
        print("Iteration ",j," solved in ",steps)
        
    
# Let's play
print("Let's play")
Games = np.zeros(100)
for j in range(100):
    state = Game.reset()
    Game.initialise = False
    done = False
    statelist = []
    stateM = state.reshape(size,size)
    canonical = Game.findcanonical(stateM)
    statelist.append(canonical)
    while done == False and steps < 100:
        x_val = state.reshape((1,size**2))
        action = ffsn_multi.predict(x_val)
        action1 = np.argmax(action, 0)
        step = Game.step(action1)
        k = 0
        canonical = Game.findcanonical(step[0].reshape(size,size))
        while canonical in statelist and k < 4*(size-1):
            action1 = action.argsort()[-k]
            Game.state = state
            step = Game.step(action1)
            canonical = Game.findcanonical(step[0].reshape(size,size))
            k += 1
        if k == 4*(size-1):
            step = Game.step(np.argmax(action))
        steps += 1
        done = step[2]
        state = step[0]
    Games[j] = steps
    print("Iteration ",j," solved in ",steps)