# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 11:12:16 2020

@author: thorstens
"""

import numpy as np
from WrapSlide import Wrapslide
from MultiClassNetwork import FFSN_MultiClass

def EvaluateObjective(ffsn):
    Game = Wrapslide()
    size = Game.size
    colours = Game.colours
    actions = 4*(size-1)
    state = Game.reset()
    done = False
    steps = 0
    while done == False and steps < 100:
        x_val = state.reshape((1,size**2))
        action = ffsn.predict(x_val)
        action = np.argmax(action, 0)
        step = Game.step(action)
        steps += 1
        done = step[2]
        state = step[0]
    return steps

Results = np.zeros(30)
Game = Wrapslide()
size = Game.size
colours = Game.colours
actions = 4*(size-1)
state = Game.reset()
done = False
steps = 0
n_input = size**2
n_output = actions
n_hidden = [6]
for i in range(30):
    print(i)
    ffsn = FFSN_MultiClass(n_input, n_output, n_hidden)
    Results[i] = EvaluateObjective(ffsn)