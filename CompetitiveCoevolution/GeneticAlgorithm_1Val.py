# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:25:24 2020

@author: thorstens
"""
from FFSNeuralNetwork import FFSNNetwork
from MultiClassNetwork import FFSN_MultiClass
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import operator
from WrapSlide import Wrapslide
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

def GeneratePopulation(size, n_input, n_output, n_hidden):
    for i in range(size):
        # Initialise a new random neural network
        ffsn_multi = FFSN_MultiClass(n_input, n_output, n_hidden)
        # Recover the weights and bias from the random network
        W = ffsn_multi.W
        B = ffsn_multi.B
        # Convert the dictionary values to vectors
        weights = dict2weights(W)
        bias = dict2weights(B)
        # Concatenate the weight vectors into a particle
        chromosome = np.concatenate((weights, bias))
        # Combine the particles into a swarm
        if i == 0:
            Population = chromosome
        else:
            Population = np.concatenate((Population,chromosome))
    # Reshape the swarm to the right output format
    Population = Population.reshape((size,(len(weights)+len(bias))))
    return Population


def EvaluateObjective(Swarm, weights, bias, n_input, n_output, n_hidden):
    Evaluations = 1
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
            Wupdated = weights2dict(NNweights, n_input, n_hidden)
            Bupdated = bias2dict(NNbias, n_hidden)
            # Update the values within the neural network
            ffsn_multi.W = Wupdated
            ffsn_multi.B = Bupdated
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
                    values[i] = ffsn_multi.predict(x_val)
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
            ffsn_multi.W = Wupdated
            ffsn_multi.B = Bupdated     
            # Perform the prediction on the train set
            Results[i, j] = PlayGame(ffsn_multi, state)
    for j in range(len(Swarm)):
        ObjValue[j] = np.mean(Results[j, :])
    return ObjValue


def PlayGame(ffsn, state_in):
    MaxSteps = Game.level*multiplier
    steps = 0
    done = False
    values = np.zeros(4*(Game.size-1))
    state = state_in
    statelist = []
    stateM = state.reshape(size,size)
    canonical = Game.findcanonical(stateM)
    statelist.append(canonical)
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
        k = 0
        canonical = Game.findcanonical(step[0].reshape(size,size))
        while canonical in statelist and k < 4*(size-1):
            action = values.argsort()[-k]
            Game.state = state
            step = Game.step(action)
            canonical = Game.findcanonical(step[0].reshape(size,size))
            k += 1
        if k == 4*(size-1):
            step = Game.step(np.argmax(values))
        steps += 1
        done = step[2]
        state = step[0]
        statelist.append(canonical)
    return (MaxSteps - steps)

def GeneticAlgorithm(n_input, n_output, n_hidden, first, previous, PopSize):
    mutation_gene_percentage = 0.2
    crossover_percentage = 0.75
    elitist_percentage = 0.2
    length = 0
    weights = 0
    bias = 0
    architecture = np.concatenate(([n_input], n_hidden, [n_output]))
    for i in range(len(architecture)-1):
        weights = weights + architecture[i]*architecture[i+1]
        bias = bias + architecture[i+1]
    length = weights + bias
    Generations = 500
    if first == True:
        Population = GeneratePopulation(PopSize, n_input, n_output, n_hidden)
        NewGeneration = Population
    else:
        Population = previous
        NewGeneration = previous
    Fitness = Objective(Population, weights, bias, n_input, n_output, n_hidden)
    Best = Population[Fitness.argsort()[-(i + 1)],:]
    MaxFitness = max(Fitness)
    print(Fitness)
    iterations = 0
    Tracker = 50
    NoChange = 0
    CumSelectionProb = np.zeros(PopSize)
    MaxFitness = max(Fitness)
    while iterations < Generations and NoChange < Tracker:
        Fitness_Sum = np.sum(Fitness)
        if Fitness_Sum == 0:
            SelectionProb = np.ones(PopSize)/PopSize
        else:
            SelectionProb = Fitness/Fitness_Sum
        for i in range(PopSize):
            if i == 0:
                CumSelectionProb[i] = SelectionProb[i]
            else:
                CumSelectionProb[i] = CumSelectionProb[i-1] + SelectionProb[i]
        
        # Perform elitism
        Counter = 0
        for i in range(int(elitist_percentage*PopSize)):
            NewGeneration[i, :] = Population[Fitness.argsort()[-(i + 1)],:]
            Counter += 1
        # Perform crossover
        for i in range(int(crossover_percentage*PopSize)):
            r1 = np.random.random()
            r2 = np.random.random()
            Parent1 = Population[np.searchsorted(CumSelectionProb,r1,side = 'left'), :]
            Parent2 = Population[np.searchsorted(CumSelectionProb,r2,side = 'left'), :]
            CrossOverpoint = np.random.randint(low = 0, high = length)
            NewGeneration[Counter,0:CrossOverpoint] = Parent1[0:CrossOverpoint]
            NewGeneration[Counter,CrossOverpoint:length] = Parent2[CrossOverpoint:length]
            Counter += 1
        # Perform mutation
        for i in range(int(PopSize-Counter)):
            mutated_values = np.random.randn(int(mutation_gene_percentage*length))
            mutated_chromosomes = np.random.randint(0,length, int(mutation_gene_percentage*length))
            r1 = np.random.random()
            NewGeneration[Counter, :] = Population[np.searchsorted(CumSelectionProb,r1,side = 'left'), :]
            for j in range(int(mutation_gene_percentage*length)):
                NewGeneration[Counter, mutated_chromosomes[j]] = mutated_values[j]
            Counter += 1
        Population = NewGeneration
        Fitness = Objective(Population, weights, bias, n_input, n_output, n_hidden)
        print(Fitness)
        if max(Fitness) > MaxFitness:
            MaxFitness = max(Fitness)
            Best = Population[Fitness.argsort()[-(i + 1)],:]
            NoChange = 0
        else:
            NoChange += 1
        iterations += 1
    return Fitness, Population, Best    

# hidden sizes = [layer 1 size, layer 2 size, etc]
Game = Wrapslide()
size = Game.size
colours = Game.colours
actions = 4*(size-1)
state = Game.reset()
done = False
steps = 0
n_input = size**2
n_output = 1
n_hidden = [20]
PopSize = 20
Game.level = 1
#pool = Pool(cpu_count())
multiplier = 5

# Generate the NN and its associated structure
ffsn_multi = FFSNNetwork(n_input, n_hidden)

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

first = True

previous = np.zeros((PopSize,(len(weights)+len(bias))))
Fitness, Population, OptimalNetwork = GeneticAlgorithm(n_input, n_output, n_hidden, first, previous, PopSize)
#OptimalNetwork = Population[Fitness.argsort()[-1],:]
print(Population)

np.savetxt("populations/population_{}multiplier_{}hidden_{}Level.csv".format(multiplier,n_hidden,Game.level), Population, delimiter=",")

# Break up the GBest particle into weight and bias components
weights = OptimalNetwork[0:len(weights)]
bias = OptimalNetwork[len(weights):len(particle)]
    
Wupdated = weights2dict(weights, n_input, n_hidden)
Bupdated = bias2dict(bias, n_hidden)
    
ffsn_multi.W = Wupdated
ffsn_multi.B = Bupdated

np.savetxt("populations/population_{}multiplier_{}hidden_{}Level.csv".format(multiplier,n_hidden,Game.level), Population, delimiter=",")

# Let's play
print("Let's play")
Games = np.zeros(100)
for j in range(100):
    state = Game.reset()
    #print(state)
    done = False
    steps = 0
    values = np.zeros(4*(size-1))
    while done == False and steps < 20:
        x_val = state.reshape((1, size**2))
        for i in range(actions):
            Game.state = state
            step = Game.step(i)
            done = step[2]
            stateNew = step[0]
            x_val = stateNew.reshape((1, size**2))
            values[i] = ffsn_multi.predict(x_val)
        action = np.argmax(values)
        Game.state = state
        step = Game.step(action)
        steps += 1
        done = step[2]
        state = step[0]
    Games[j] = steps
    print("Iteration ", j, " solved in ", steps)

first = False

for i in range(5):
    Game.level = i + 2
    # Let the optimisation begin
    print(Population)
    Fitness, Population, OptimalNetwork = GeneticAlgorithm(n_input, n_output, n_hidden, first, previous, PopSize)
    #OptimalNetwork = Population[Fitness.argsort()[-1],:]
    print(Population)

    np.savetxt("populations/population_{}multiplier_{}hidden_{}Level.csv".format(multiplier,n_hidden,Game.level), Population, delimiter=",")
    # Break up the GBest particle into weight and bias components
    weights = OptimalNetwork[0:len(weights)]
    bias = OptimalNetwork[len(weights):len(particle)]
    
    Wupdated = weights2dict(weights, n_input, n_hidden)
    Bupdated = bias2dict(bias, n_hidden)
    
    ffsn_multi.W = Wupdated
    ffsn_multi.B = Bupdated
    
    # Let's play
    print("Let's play")
    Games = np.zeros(100)
    for j in range(100):
        state = Game.reset()
        statelist = []
        stateM = state.reshape(size,size)
        canonical = Game.findcanonical(stateM)
        statelist.append(canonical)
        #print(state.reshape(size,size))
        done = False
        steps = 0
        while done == False and steps < 100:
            x_val = state.reshape((1, size**2))
            for i in range(actions):
                Game.state = state
                step = Game.step(i)
                done = step[2]
                stateNew = step[0]
                x_val = stateNew.reshape((1, size**2))
                values[i] = ffsn_multi.predict(x_val)
            action = np.argmax(values)
            Game.state = state
            step = Game.step(action)
            k = 0
            canonical = Game.findcanonical(step[0].reshape(size,size))
            while canonical in statelist and k < 4*(size-1):
                action = values.argsort()[-k]
                #print("Action ", action, k)
                Game.state = state
                step = Game.step(action)
                canonical = Game.findcanonical(step[0].reshape(size,size))
                k += 1
            if k == 4*(size-1):
                step = Game.step(np.argmax(values))    
            steps += 1
            done = step[2]
            state = step[0]
            statelist.append(canonical)
        Games[j] = steps
        print("Iteration ", j, " solved in ", steps)



Game.initialise = False
# Let's play
print("Let's play")
Games = np.zeros(100)
for j in range(100):
    state = Game.reset()
    statelist = []
    stateM = state.reshape(size,size)
    canonical = Game.findcanonical(stateM)
    statelist.append(canonical)
    #print(state.reshape(size,size))
    done = False
    steps = 0
    while done == False and steps < 1000:
        x_val = state.reshape((1, size**2))
        for i in range(actions):
            Game.state = state
            step = Game.step(i)
            done = step[2]
            stateNew = step[0]
            x_val = stateNew.reshape((1, size**2))
            values[i] = ffsn_multi.predict(x_val)
        action = np.argmax(values)
        Game.state = state
        step = Game.step(action)
        k = 0
        canonical = Game.findcanonical(step[0].reshape(size,size))
        while canonical in statelist and k < 4*(size-1):
            action = values.argsort()[-k]
            #print("Action ", action, k)
            Game.state = state
            step = Game.step(action)
            canonical = Game.findcanonical(step[0].reshape(size,size))
            k += 1
        if k == 4*(size-1):
            step = Game.step(np.argmax(values))    
        steps += 1
        done = step[2]
        state = step[0]
        statelist.append(canonical)
    Games[j] = steps
    print("Iteration ", j, " solved in ", steps)