#JE KNOLL & T Schmidt-Dumont
#Wrapslide Environment V0


import gym
from gym import error, spaces, utils
from gym.utils import seeding
from bitstring import BitStream
import numpy as np
import random
import copy
import math
 
class WrapslideEnv(gym.Env):  
    #metadata = {'render.modes': ['human']}
    
    def __init__(self):
        
        #Define the size of the grid and the number of colours here.
        self.size = 6
        self.colours = 3
        self.initialise = False
        self.level = 6
        self.test = False
        self.Convnet = True
        
        high = np.repeat(4, 4*self.size)
        low = np.repeat(0, 4*self.size)
        
        self.action_space = spaces.Discrete(4*(self.size-1))
        #self.observation_space = spaces.Box(low,high,dtype=np.float32)
        self.observation_space = spaces.Box(low=1,high=4, shape=(self.size, self.size,1), dtype=np.uint8)
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        
        self.stateList = []
        
        if self.colours == 2:
            x = np.array([[1,1]])
            solved_state = np.repeat(x,self.size/2,1)
            for i in range(int(self.size/2)-1):
                y = np.repeat(x,self.size/2,1)
                solved_state = np.vstack((solved_state,y))
            x = np.array([[1,2]])
            for i in range(int(self.size/2)):
                y = np.repeat(x,self.size/2,1)
                solved_state = np.vstack((solved_state,y))
                
        if self.colours == 3:
            x = np.array([[1,1]])
            solved_state = np.repeat(x,self.size/2,1)
            for i in range(int(self.size/2)-1):
                y = np.repeat(x,self.size/2,1)
                solved_state = np.vstack((solved_state,y))
            x = np.array([[2,3]])
            for i in range(int(self.size/2)):
                y = np.repeat(x,self.size/2,1)
                solved_state = np.vstack((solved_state,y))
                
        if self.colours == 4:
            x = np.array([[1,2]])
            solved_state = np.repeat(x,self.size/2,1)
            for i in range(int(self.size/2)-1):
                y = np.repeat(x,self.size/2,1)
                solved_state = np.vstack((solved_state,y))
            x = np.array([[3,4]])
            for i in range(int(self.size/2)):
                y = np.repeat(x,self.size/2,1)
                solved_state = np.vstack((solved_state,y))
        
        self.doneState = self.findcanonical(solved_state)
        
        if self.initialise == True:
            Root = solved_state
            #Generate the bottom part of the tree
            print("This is the root \n", Root)
            # Level 1
            Level1 = []
            Level1CanonicalB = []
            n = self.size
            for i in range(0,n-1):
                #if i == 2:
                    #i = 3
                grid = self.move_top_half(Root,n-1-i)
                Level1.append(grid)
                grid = self.move_bottom_half(Root,n-1-i)
                Level1.append(grid)
                grid = self.move_left_half(Root,n-1-i)
                Level1.append(grid)
                grid = self.move_right_half(Root,n-1-i)
                Level1.append(grid)
            
            for i in range(len(Level1)):
                Level1CanonicalB.append(self.findcanonical(Level1[i]))
            
            Level1CanonicalB = np.array(Level1CanonicalB)
            Level1CanonicalB = np.unique(Level1CanonicalB)
            
            Level1 = []
            FromBottom = []
            for i in range(len(Level1CanonicalB)):
                Level1.append(self.CanonicalToGrid(Level1CanonicalB[i]))
                FromBottom.append(Level1CanonicalB[i])
            self.Level1UniqueB = Level1CanonicalB
            
            if self.level > 1:    
                # Level 2
                Level2 = []
                Level2Canonical = []
                for i in range(len(Level1)):
                    for j in range(n-1):
                        #if j == 2:
                        #   j = 3
                        grid = self.move_top_half(Level1[i],n-1-j)
                        Level2.append(grid)
                        grid = self.move_bottom_half(Level1[i],n-1-j)
                        Level2.append(grid)
                        grid = self.move_left_half(Level1[i],n-1-j)
                        Level2.append(grid)
                        grid = self.move_right_half(Level1[i],n-1-j)
                        Level2.append(grid)
                    
                for i in range(len(Level2)):
                    Level2Canonical.append(self.findcanonical(Level2[i]))
                
                Level2UniqueB, indecies2 = np.unique(Level2Canonical, return_index = True)
                Level2 = []
                for i in range(len(Level2UniqueB)):
                    Level2.append(self.CanonicalToGrid(int(Level2UniqueB[i])))
                    FromBottom.append(Level2UniqueB[i])
                self.Level2UniqueB = Level2UniqueB
            
            if self.level > 2:    
                # Level 3
                Level3 = []
                Level3Canonical = []
                for i in range(len(Level2)):
                    for j in range(n-1):
                        #if j == 2:
                        #   j = 3
                        grid = self.move_top_half(Level2[i],n-1-j)
                        Level3.append(grid)
                        grid = self.move_bottom_half(Level2[i],n-1-j)
                        Level3.append(grid)
                        grid = self.move_left_half(Level2[i],n-1-j)
                        Level3.append(grid)
                        grid = self.move_right_half(Level2[i],n-1-j)
                        Level3.append(grid)
                    
                for i in range(len(Level3)):
                    Level3Canonical.append(self.findcanonical(Level3[i]))
                
                Level3UniqueB, indecies3 = np.unique(Level3Canonical, return_index = True)
                Level3 = []
                for i in range(len(Level3UniqueB)):
                    Level3.append(self.CanonicalToGrid(int(Level3UniqueB[i])))
                    FromBottom.append(Level3UniqueB[i])
                self.Level3UniqueB = Level3UniqueB
    
            
            if self.level > 3:
                # Level 4
                Level4 = []
                Level4Canonical = []
                for i in range(len(Level3)):
                    for j in range(n-1):
                        if j == 2:
                           j = 3
                        grid = self.move_top_half(Level3[i],n-1-j)
                        Level4.append(grid)
                        grid = self.move_bottom_half(Level3[i],n-1-j)
                        Level4.append(grid)
                        grid = self.move_left_half(Level3[i],n-1-j)
                        Level4.append(grid)
                        grid = self.move_right_half(Level3[i],n-1-j)
                        Level4.append(grid)
                    
                for i in range(len(Level4)):
                    Level4Canonical.append(self.findcanonical(Level4[i]))
                
                Level4UniqueB, indecies4 = np.unique(Level4Canonical, return_index = True)
                Level4 = []
                for i in range(len(Level4UniqueB)):
                    Level4.append(self.CanonicalToGrid(int(Level4UniqueB[i])))
                    FromBottom.append(Level4UniqueB[i])
                self.Level4UniqueB = Level4UniqueB
            
            if self.level > 4:
                # Level 5
                Level5 = []
                Level5Canonical = []
                for i in range(len(Level4)):
                    for j in range(n-1):
                        if j == 2:
                           j = 3
                        grid = self.move_top_half(Level4[i],n-1-j)
                        Level5.append(grid)
                        grid = self.move_bottom_half(Level4[i],n-1-j)
                        Level5.append(grid)
                        grid = self.move_left_half(Level4[i],n-1-j)
                        Level5.append(grid)
                        grid = self.move_right_half(Level4[i],n-1-j)
                        Level5.append(grid)
                
                for i in range(len(Level5)):
                    Level5Canonical.append(self.findcanonical(Level5[i]))
                
                Level5UniqueB, indecies5 = np.unique(Level5Canonical, return_index = True)
                Level5 = []
                for i in range(len(Level5UniqueB)):
                    Level5.append(self.CanonicalToGrid(int(Level5UniqueB[i])))
                    FromBottom.append(Level5UniqueB[i])
                self.Level5UniqueB = Level5UniqueB
            
            if self.level > 5:
                # Level 6
                Level6 = []
                Level6Canonical = []
                for i in range(len(Level5)):
                    for j in range(n-1):
                        if j == 2:
                           j = 3
                        grid = self.move_top_half(Level5[i],n-1-j)
                        Level6.append(grid)
                        grid = self.move_bottom_half(Level5[i],n-1-j)
                        Level6.append(grid)
                        grid = self.move_left_half(Level5[i],n-1-j)
                        Level6.append(grid)
                        grid = self.move_right_half(Level5[i],n-1-j)
                        Level6.append(grid)
                
                for i in range(len(Level6)):
                    Level6Canonical.append(self.findcanonical(Level6[i]))
                
                Level6UniqueB, indecies6 = np.unique(Level6Canonical, return_index = True)
                Level6 = []
                for i in range(len(Level6UniqueB)):
                    Level6.append(self.CanonicalToGrid(int(Level6UniqueB[i])))
                    FromBottom.append(Level6UniqueB[i])
                self.Level6UniqueB = Level6UniqueB

   
    def step(self, action):
        '''
        Action Space = Discrete(8):
        0 = left half up 'lhu'
        1 = left half down 'lhd'
        2 = right half up 'rhu'
        3 = right half down 'rhd'
        4 = top half left 'thl'
        5 = top half right 'thr'
        6 = bottom half left 'bhl'
        7 = bottom half down 'bhr'
        '''
        #self.DIRECTIONS = ['lhu', 'lhd', 'rhu', 'rhd', 'thl', 'thr', 'bhl', 'bhr']
        #print('begin step. State is:')
        if self.Convnet == False:
            self.state = self.bitstring_to_grid(self.state)
        #print('State after transformed to grid and before step:')

        #print('')
        stateM = self.state.reshape(self.size,self.size)
        #print(stateM)
        slide = action
        #Only perform this step if testing == true
        if self.test == True:
            slide = np.argmax(action)
        #First its gonna do the action that was chosen by the agent.

        #print("Before the move the state is\n ",stateM)

        x = math.floor(slide/(self.size-1))
        y = slide % (self.size-1)
        if x == 0:
            if y == 0:
                stateM = self.move_left_half(stateM, 1)
            elif y == 1:
                stateM = self.move_left_half(stateM, -1)
            elif y == 2:
                stateM = self.move_left_half(stateM, 2)
            elif y == 3:
                stateM = self.move_left_half(stateM, -2)
            elif y == 4:
                stateM = self.move_left_half(stateM, 3)
            elif y == 5:
                stateM = self.move_left_half(stateM, -3)
            elif y == 6:
                stateM = self.move_left_half(stateM, 4)
        elif x == 1:
            if y == 0:
                stateM = self.move_right_half(stateM, 1)
            elif y == 1:
                stateM = self.move_right_half(stateM, -1)
            elif y == 2:
                stateM = self.move_right_half(stateM, 2)
            elif y == 3:
                stateM = self.move_right_half(stateM, -2)
            elif y == 4:
                stateM = self.move_right_half(stateM, 3)
            elif y == 5:
                stateM = self.move_right_half(stateM, -3)
            elif y == 6:
                stateM = self.move_right_half(stateM, 4)
        elif x == 2:
            if y == 0:
                stateM = self.move_top_half(stateM, 1)
            elif y == 1:
                stateM = self.move_top_half(stateM, -1)
            elif y == 2:
                stateM = self.move_top_half(stateM, 2)
            elif y == 3:
                stateM = self.move_top_half(stateM, -2)
            elif y == 4:
                stateM = self.move_top_half(stateM, 3)
            elif y == 5:
                stateM = self.move_top_half(stateM, -3)
            elif y == 6:
                stateM = self.move_top_half(stateM, 4)
        elif x == 3:
            if y == 0:
                stateM = self.move_bottom_half(stateM, 1)
            elif y == 1:
                stateM = self.move_bottom_half(stateM, -1)
            elif y == 2:
                stateM = self.move_bottom_half(stateM, 2)
            elif y == 3:
                stateM = self.move_bottom_half(stateM, -2)
            elif y == 4:
                stateM = self.move_bottom_half(stateM, 3)
            elif y == 5:
                stateM = self.move_bottom_half(stateM, -3)
            elif y == 6:
                stateM = self.move_bottom_half(stateM, 4)

        #print('After Step is completed, state is: \n', stateM)
        #print(self.state)
        canonical = self.findcanonical(stateM)
        
        #Only perform this step if testing == true
        if self.test == True:
            i = 1
            while canonical in self.stateList:
                #print(action.argsort()[-i])
                stateM = self.random_action(action.argsort()[-i])
                canonical = self.findcanonical(stateM)
                i = i + 1
        
        if canonical == self.doneState:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        self.stateList.append(canonical)
        self.state = self.CanonicalToGrid(canonical)

        state1 = np.zeros(self.size**2)
        
        for i in range(self.size):
            for j in range(self.size):
                state1[i*self.size+j] = self.state[i,j]
        
        bits = self.convert_to_bit(state1)
        self.steps_beyond_done = None
        # Comment out line below if convnet is used.
        if self.Convnet == True:
            self.state = self.state
        else:
            self.state = state1
            
        return self.state,  reward, done, {}   
            
   
    def reset(self):
        if self.colours == 2:
            state = self.generate_two(self.size)
        elif self.colours == 3:
            if self.initialise == False:
                state = self.generate_threeColour(self.size)
            else:
                state = self.generate_threeColour_initialise(self.size, self.level)
        else:
            if self.initialise == False:
                state = self.generate_fourColour(self.size)
            else:
                state = self.generate_fourColour_initialise(self.size, self.level)
        state = state.astype(int)
        state1 = np.zeros(self.size**2)
        #print("This is the starting state \n", state)
        for i in range(self.size):
            for j in range(self.size):
                state1[i*self.size+j] = state[i,j]
        
        bits = self.convert_to_bit(state1)
        self.steps_beyond_done = None
        self.stateList = []
        #self.state = bits.bin
        if self.Convnet == True:
            self.state = state
        else:
            self.state = state1
        #self.state = self.state.reshape(4,4)
        #print("This is the start:\n", self.state)
        return self.state

    

    def bitstring_to_grid(self, bitstring):
        togrid = bitstring[0:]
        state = np.zeros((self.size, self.size))
        k=0
        for i in range(self.size):
            for j in range(self.size):
                state[i,j] = togrid[k]
                k=k+1 
        return state.astype(int)
    
    def random_action(self, action):
        
        #rand_action = random.randrange(4*(self.size-1))
        rand_action = action
        stateM = self.state
        action = rand_action
        x = math.floor(action/(self.size-1))
        y = action % (self.size-1)

        if x == 0:
            if y == 0:
                stateM = self.move_left_half(stateM, 1)
            elif y == 1:
                stateM = self.move_left_half(stateM, -1)
            elif y == 2:
                stateM = self.move_left_half(stateM, 2)
            elif y == 3:
                stateM = self.move_left_half(stateM, -2)
            elif y == 4:
                stateM = self.move_left_half(stateM, 3)
            elif y == 5:
                stateM = self.move_left_half(stateM, -3)
            elif y == 6:
                stateM = self.move_left_half(stateM, 4)
        elif x == 1:
            if y == 0:
                stateM = self.move_right_half(stateM, 1)
            elif y == 1:
                stateM = self.move_right_half(stateM, -1)
            elif y == 2:
                stateM = self.move_right_half(stateM, 2)
            elif y == 3:
                stateM = self.move_right_half(stateM, -2)
            elif y == 4:
                stateM = self.move_right_half(stateM, 3)
            elif y == 5:
                stateM = self.move_right_half(stateM, -3)
            elif y == 6:
                stateM = self.move_right_half(stateM, 4)
        elif x == 2:
            if y == 0:
                stateM = self.move_top_half(stateM, 1)
            elif y == 1:
                stateM = self.move_top_half(stateM, -1)
            elif y == 2:
                stateM = self.move_top_half(stateM, 2)
            elif y == 3:
                stateM = self.move_top_half(stateM, -2)
            elif y == 4:
                stateM = self.move_top_half(stateM, 3)
            elif y == 5:
                stateM = self.move_top_half(stateM, -3)
            elif y == 6:
                stateM = self.move_top_half(stateM, 4)
        elif x == 3:
            if y == 0:
                stateM = self.move_bottom_half(stateM, 1)
            elif y == 1:
                stateM = self.move_bottom_half(stateM, -1)
            elif y == 2:
                stateM = self.move_bottom_half(stateM, 2)
            elif y == 3:
                stateM = self.move_bottom_half(stateM, -2)
            elif y == 4:
                stateM = self.move_bottom_half(stateM, 3)
            elif y == 5:
                stateM = self.move_bottom_half(stateM, -3)
            elif y == 6:
                stateM = self.move_bottom_half(stateM, 4)
        return stateM
        
    def generate_two(self, n):
        one=0
        two=0
    
    
        initial=np.zeros((n,n))
        for i in range(0,n):
            for j in range(0,n):
                while initial[i][j]==0:
                    rand = random.randrange(n) + 1
                    #print(rand)
                    if (rand==1 and (one<int(((n/2)**2)*3))):
                            initial[i][j] = rand
                            one=one+1
    
                    # elif one==int((n/2)**2):
                    #     while rand==1:
                    #         rand = random.randrange(n) + 1
    
                    if (rand == 2 and (two < int(((n/2)**2)))):
                            initial[i][j] = rand
                            two = two + 1
    
        return initial
    
    def generate_threeColour(self, n):
        one=0
        two=0
        three=0
    
        initial=np.zeros((n,n))
        for i in range(0,n):
            for j in range(0,n):
                while initial[i][j]==0:
                    rand = random.randrange(n) + 1
                    if (rand==1 and (one<int(((n/2)**2)*2))):
                            initial[i][j] = rand
                            one=one+1

    
                    if (rand == 2 and (two < int(((n/2)**2)))):
                            initial[i][j] = rand
                            two = two + 1
    
                    if (rand == 3 and (three < int(((n / 2) ** 2) ))):
                            initial[i][j] = rand
                            three = three + 1
        return initial
    
    def generate_threeColour_initialise(self, n, level):
        if level == 1:
            rand = random.randint(0,len(self.Level1UniqueB)-1)
            canonical = self.Level1UniqueB[rand]
        elif level == 2:
            rand = random.randint(0,len(self.Level2UniqueB)-1)
            canonical = self.Level2UniqueB[rand]
        elif level == 3:
            rand = random.randint(0,len(self.Level3UniqueB)-1)
            canonical = self.Level3UniqueB[rand]
        elif level == 4:
            rand = random.randint(0,len(self.Level4UniqueB)-1)
            canonical = self.Level4UniqueB[rand]
        elif level == 5:
            rand = random.randint(0,len(self.Level5UniqueB)-1)
            canonical = self.Level5UniqueB[rand]
        elif level == 6:
            rand = random.randint(0,len(self.Level6UniqueB)-1)
            canonical = self.Level6UniqueB[rand]
        initial = self.CanonicalToGrid(canonical)
        return initial

    def generate_fourColour(self, n):
        one=0
        two=0
        three=0
        four=0
        initial=np.zeros((n,n))
        for i in range(0,n):
            for j in range(0,n):
                while initial[i][j]==0:
                    rand = random.randrange(n) + 1
                    if (rand==1 and (one<int((n/2)**2))):
                            initial[i][j] = rand
                            one=one+1
    
                    if (rand == 2 and (two < int(((n/2)**2)))):
                            initial[i][j] = rand
                            two = two + 1
    
                    if (rand == 3 and (three < int(((n / 2) ** 2) ))):
                            initial[i][j] = rand
                            three = three + 1
    
                    if (rand == 4 and (four < int((n / 2) ** 2 ))):
                            initial[i][j] = rand
                            four = four + 1

        return initial
    
    def generate_fourColour_initialise(self, n, level):
        if level == 1:
            rand = random.randint(0,len(self.Level1UniqueB)-1)
            canonical = self.Level1UniqueB[rand]
        elif level == 2:
            rand = random.randint(0,len(self.Level2UniqueB)-1)
            canonical = self.Level2UniqueB[rand]
        elif level == 3:
            rand = random.randint(0,len(self.Level3UniqueB)-1)
            canonical = self.Level3UniqueB[rand]
        elif level == 4:
            rand = random.randint(0,len(self.Level4UniqueB)-1)
            canonical = self.Level4UniqueB[rand]
        elif level == 5:
            rand = random.randint(0,len(self.Level5UniqueB)-1)
            canonical = self.Level5UniqueB[rand]
        elif level == 6:
            rand = random.randint(0,len(self.Level6UniqueB)-1)
            canonical = self.Level6UniqueB[rand]
        initial = self.CanonicalToGrid(canonical)
        return initial

#Convert array of state (grid) into a bitstream
    def convert_to_bit(self, state_array):
        if self.size == 4:
            bits = BitStream(bin='00000000000000000000000000000000')
        elif self.size == 6:
            bits = BitStream(bin='000000000000000000000000000000000000000000000000000000000000000000000000')
        elif self.size == 8:
            bits = BitStream(bin='00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000')
        for i in range(0, self.size**2):
            if state_array[i] == 1:
                bits[2 * i] = 0
                bits[2 * i + 1] = 0
            elif state_array[i] == 2:
                bits[2 * i] = 0
                bits[2 * i + 1] = 1
            elif state_array[i] == 3:
                bits[2 * i] = 1
                bits[2 * i + 1] = 0
            elif state_array[i] == 4:
                bits[2 * i] = 1
                bits[2 * i + 1] = 1
            i = i + 1
        return bits
    
    def threeD_to_lists(self, isomorphs):
        i = 0
        tempcanonical = isomorphs
        for i in range(0, 32):
            tempcanonical[:, :, i] = self.colourswop(isomorphs[:, :, i])
            i = i + 1
    
        j = 0
        colourlist = [[]] * 32
        for j in range(0, 32):
            colourlist[j] = tempcanonical[:, :, j].tolist()
    
        gridlists = [[]] * 32
        for i in range(0, 32):
            gridlists[i] = np.ravel(colourlist[i])
            i = i + 1
        return gridlists

        
        
    def move_top_half(self, grid, n):
        # roll top half n blocks: positive n rolls to right negative n rolls left
        gridnew = copy.deepcopy(grid)
        for i in range(int(self.size/2)):
            gridnew[i] = np.roll(gridnew[i], n)
        return gridnew
    
    def move_bottom_half(self, grid, n):
        gridnew = copy.deepcopy(grid)
        for i in range(int(self.size/2)):
            gridnew[i+int(self.size/2)] = np.roll(gridnew[i+int(self.size/2)], n)
        return gridnew
    
    def move_left_half(self, grid, n):
        gridtemp = copy.deepcopy(grid)
        gridnew = np.transpose(gridtemp)
        for i in range(int(self.size/2)):
            gridnew[i] = np.roll(gridnew[i], n)
        #gridnew[1] = np.roll(gridnew[1], n)
        return np.transpose(gridnew)
    
    def move_right_half(self, grid, n):
        gridtemp = copy.deepcopy(grid)
        gridnew = np.transpose(gridtemp)
        for i in range(int(self.size/2)):
            gridnew[i+int(self.size/2)] = np.roll(gridnew[i+int(self.size/2)], n)
        #gridnew[3] = np.roll(gridnew[3], n)
        return np.transpose(gridnew)
    
    
    # Rotate original matrix 90 degrees anticlockwise 3 times resulting in four isomorphs total
    def rotate(self, grid):
        iso1 = grid
        iso2 = np.rot90(grid, k=1)
        iso3 = np.rot90(grid, k=2)
        iso4 = np.rot90(grid, k=3)
        rotations = np.dstack((iso1, iso2, iso3, iso4))
        return rotations
    
    
    # Isomorph 1 centre translations
    def centre_translations(self, grid):
        n = int(self.size/2)
        iso1a = grid
        iso1b = np.roll(grid, n, axis=0)
        iso1c = np.roll(grid, n, axis=1)
        iso1d = np.roll(iso1c, n, axis=0)
        centretranslations = np.dstack((iso1a, iso1b, iso1c, iso1d))
        return centretranslations
    
    
    # Flips
    def fliparray(self, grid):
        iso1ai = grid
        iso1bii = np.fliplr(grid)
        flipped = np.dstack((iso1ai, iso1bii))
        return flipped
    
    
    # send a 2D array to find colour combo that will result in smallest decimal value: HHHHHHHHHH hierdie sal moet verander as nomering verander
    
    def colourswop(self, grid):
        gridnew = grid
    
        if gridnew[0, 0] != 1:
            k = gridnew[0, 0]
            np.place(gridnew, gridnew == 1, [11])
            np.place(gridnew, gridnew == k, [1])
            np.place(gridnew, gridnew == 11, [k])
            # print(gridnew)
    
        t = 0
        v = 1
        maxcol = 1
    
        for t in range(0, self.size):
            for v in range(0, self.size):
                if gridnew[t, v] > maxcol:
                    # print(maxcol)
                    k = gridnew[t, v]
                    h = maxcol + 1
                    if h != k:
                        np.place(gridnew, gridnew == h, [11])
                        np.place(gridnew, gridnew == k, [h])
                        np.place(gridnew, gridnew == 11, [k])
                        # print("next",t,v)
                        # print(gridnew)
                    maxcol = h
    
                v = v + 1
    
            t = t + 1
        return gridnew
    
    
    def convert_to_smallestint(self, colours):
        isoints = [0] * 32
        #print("This is colours\n", colours)
        j = 0
        for j in range(0, 32):
            i = 0
            if self.size == 4:
                bits = BitStream(bin='00000000000000000000000000000000')
            elif self.size == 6:
                bits = BitStream(bin='000000000000000000000000000000000000000000000000000000000000000000000000')
            elif self.size == 8:
                bits = BitStream(bin='00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000')
            for i in range(0, self.size**2):
                if colours[j][i] == 1:
                    bits[2 * i] = 0
                    bits[2 * i + 1] = 0
                elif colours[j][i] == 2:
                    bits[2 * i] = 0
                    bits[2 * i + 1] = 1
                elif colours[j][i] == 3:
                    bits[2 * i] = 1
                    bits[2 * i + 1] = 0
                elif colours[j][i] == 4:
                    bits[2 * i] = 1
                    bits[2 * i + 1] = 1
                i = i + 1
                isoints[j] = int(bits.bin, 2)
                # print(isoints[j])
        j = j + 1
        return isoints
    
    
    
    
    def findcanonical(self, grid):
        
        rotations = self.rotate(grid)
        iso1 = rotations[:, :, 0]
        iso2 = rotations[:, :, 1]
        iso3 = rotations[:, :, 2]
        iso4 = rotations[:, :, 3]  
    
        iso1a = (self.centre_translations(iso1))[:, :, 0]
        iso1b = (self.centre_translations(iso1))[:, :, 1]
        iso1c = (self.centre_translations(iso1))[:, :, 2]
        iso1d = (self.centre_translations(iso1))[:, :, 3]
    
        iso2a = (self.centre_translations(iso2))[:, :, 0]
        iso2b = (self.centre_translations(iso2))[:, :, 1]
        iso2c = (self.centre_translations(iso2))[:, :, 2]
        iso2d = (self.centre_translations(iso2))[:, :, 3]
    
        iso3a = (self.centre_translations(iso3))[:, :, 0]
        iso3b = (self.centre_translations(iso3))[:, :, 1]
        iso3c = (self.centre_translations(iso3))[:, :, 2]
        iso3d = (self.centre_translations(iso3))[:, :, 3]
    
        iso4a = (self.centre_translations(iso4))[:, :, 0]
        iso4b = (self.centre_translations(iso4))[:, :, 1]
        iso4c = (self.centre_translations(iso4))[:, :, 2]
        iso4d = (self.centre_translations(iso4))[:, :, 3]
    
        iso1ai = (self.fliparray(iso1a))[:, :, 0]
        iso1aii = (self.fliparray(iso1a))[:, :, 1]
    
        iso1bi = (self.fliparray(iso1b))[:, :, 0]
        iso1bii = (self.fliparray(iso1b))[:, :, 1]
    
        iso1ci = (self.fliparray(iso1c))[:, :, 0]
        iso1cii = (self.fliparray(iso1c))[:, :, 1]
    
        iso1di = (self.fliparray(iso1d))[:, :, 0]
        iso1dii = (self.fliparray(iso1d))[:, :, 1]
    
        iso2ai = (self.fliparray(iso2a))[:, :, 0]
        iso2aii = (self.fliparray(iso2a))[:, :, 1]
    
        iso2bi = (self.fliparray(iso2b))[:, :, 0]
        iso2bii = (self.fliparray(iso2b))[:, :, 1]
    
        iso2ci = (self.fliparray(iso2c))[:, :, 0]
        iso2cii = (self.fliparray(iso2c))[:, :, 1]
    
        iso2di = (self.fliparray(iso2d))[:, :, 0]
        iso2dii = (self.fliparray(iso2d))[:, :, 1]
    
        iso3ai = (self.fliparray(iso3a))[:, :, 0]
        iso3aii = (self.fliparray(iso3a))[:, :, 1]
    
        iso3bi = (self.fliparray(iso3b))[:, :, 0]
        iso3bii = (self.fliparray(iso3b))[:, :, 1]
    
        iso3ci = (self.fliparray(iso3c))[:, :, 0]
        iso3cii = (self.fliparray(iso3c))[:, :, 1]
    
        iso3di = (self.fliparray(iso3d))[:, :, 0]
        iso3dii = (self.fliparray(iso3d))[:, :, 1]
    
        iso4ai = (self.fliparray(iso4a))[:, :, 0]
        iso4aii = (self.fliparray(iso4a))[:, :, 1]
    
        iso4bi = (self.fliparray(iso4b))[:, :, 0]
        iso4bii = (self.fliparray(iso4b))[:, :, 1]
    
        iso4ci = (self.fliparray(iso4c))[:, :, 0]
        iso4cii = (self.fliparray(iso4c))[:, :, 1]
    
        iso4di = (self.fliparray(iso4d))[:, :, 0]
        iso4dii = (self.fliparray(iso4d))[:, :, 1]
    
        isomorphs = np.dstack(
            (iso1ai, iso1aii, iso1bi, iso1bii, iso1ci, iso1cii, iso1di, iso1dii, iso2ai, iso2aii, iso2bi, iso2bii, iso2ci,
             iso2cii, iso2di, iso2dii, iso3ai, iso3aii, iso3bi, iso3bii, iso3ci, iso3cii, iso3di, iso3dii, iso4ai, iso4aii,
             iso4bi, iso4bii, iso4ci, iso4cii, iso4di, iso4dii))
        
        colours = self.threeD_to_lists(isomorphs)
    
        canonical = min(self.convert_to_smallestint(colours)) #kan verander dat dit 'n 32 bit string return
    
        return canonical
    
    def CanonicalToGrid(self, canonical):
        togrid = bin(canonical)[2:].zfill(2*(self.size**2))
        state = np.zeros((self.size, self.size))
        k=0
        for i in range(0, self.size):
            j=0
            for j in range(0, self.size):
                if togrid[k * 2:k * 2 + 2] == '00':
                    state[i][j] = 1
                elif togrid[k * 2:k * 2 + 2] == '01':
                    state[i][j] = 2
                elif togrid[k * 2:k * 2 + 2] == '10':
                    state[i][j] = 3
                elif togrid[k * 2:k * 2 + 2] == '11':
                    state[i][j] = 4
                k += 1
                j += 1
            i += 1
        return state.astype(int)
    
        