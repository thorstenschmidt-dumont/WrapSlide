#JE KNOLL
#Wrapslide Environment V0


import gym
from gym import error, spaces, utils
from gym.utils import seeding
from bitstring import BitStream
import numpy as np
import random
import copy

 
class WrapslideEnv(gym.Env):  
    #metadata = {'render.modes': ['human']}
    
    def __init__(self):
        high = np.array([4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4])
        low= np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        
        self.action_space = spaces.Discrete(12) 
        self.observation_space = spaces.Box(low,high,dtype=np.float32)
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        
        self.stateList = []
        #solved_state = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 1, 1], [1, 1, 1, 1]])
        solved_state = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 1, 1], [3, 3, 1, 1]])
        #solved_state = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
        self.doneState = self.findcanonical(solved_state)
        
        """
        Root = np.array([[1,1,1,1],
                         [1,1,1,1],
                         [1,1,2,2],
                         [1,1,2,2]])
        """
        
        Root = np.array([[1,1,1,1],
                         [1,1,1,1],
                         [2,2,3,3],
                         [2,2,3,3]])
        
        """
        Root = np.array([[1,1,2,2],
                         [1,1,2,2],
                         [3,3,4,4],
                         [3,3,4,4]])
        """
        #Generate the bottom part of the tree
        #print(Root)
        # Level 1
        Level1 = []
        Level1CanonicalB = []
        n=4
        for i in range(0,n-1):
           if i == 2:
               i = 3
           grid = self.move_top_half(Root,2-i)
           Level1.append(grid)
           grid = self.move_bottom_half(Root,2-i)
           Level1.append(grid)
           grid = self.move_left_half(Root,2-i)
           Level1.append(grid)
           grid = self.move_right_half(Root,2-i)
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
            
        # Level 2
        Level2 = []
        Level2Canonical = []
        for i in range(len(Level1)):
            for j in range(n-1):
                if j == 2:
                   j = 3
                grid = self.move_top_half(Level1[i],2-j)
                Level2.append(grid)
                grid = self.move_bottom_half(Level1[i],2-j)
                Level2.append(grid)
                grid = self.move_left_half(Level1[i],2-j)
                Level2.append(grid)
                grid = self.move_right_half(Level1[i],2-j)
                Level2.append(grid)
            
        for i in range(len(Level2)):
            Level2Canonical.append(self.findcanonical(Level2[i]))
        
        Level2UniqueB, indecies2 = np.unique(Level2Canonical, return_index = True)
        Level2 = []
        for i in range(len(Level2UniqueB)):
            Level2.append(self.CanonicalToGrid(int(Level2UniqueB[i])))
            FromBottom.append(Level2UniqueB[i])
            
        # Level 3
        Level3 = []
        Level3Canonical = []
        for i in range(len(Level2)):
            for j in range(n-1):
                if j == 2:
                   j = 3
                grid = self.move_top_half(Level2[i],2-j)
                Level3.append(grid)
                grid = self.move_bottom_half(Level2[i],2-j)
                Level3.append(grid)
                grid = self.move_left_half(Level2[i],2-j)
                Level3.append(grid)
                grid = self.move_right_half(Level2[i],2-j)
                Level3.append(grid)
            
        for i in range(len(Level3)):
            Level3Canonical.append(self.findcanonical(Level3[i]))
        
        Level3UniqueB, indecies3 = np.unique(Level3Canonical, return_index = True)
        Level3 = []
        for i in range(len(Level3UniqueB)):
            Level3.append(self.CanonicalToGrid(int(Level3UniqueB[i])))
            FromBottom.append(Level3UniqueB[i])
        
        
        # Level 4
        Level4 = []
        Level4Canonical = []
        for i in range(len(Level3)):
            for j in range(n-1):
                if j == 2:
                   j = 3
                grid = self.move_top_half(Level3[i],2-j)
                Level4.append(grid)
                grid = self.move_bottom_half(Level3[i],2-j)
                Level4.append(grid)
                grid = self.move_left_half(Level3[i],2-j)
                Level4.append(grid)
                grid = self.move_right_half(Level3[i],2-j)
                Level4.append(grid)
            
        for i in range(len(Level4)):
            Level4Canonical.append(self.findcanonical(Level4[i]))
        
        Level4UniqueB, indecies4 = np.unique(Level4Canonical, return_index = True)
        Level4 = []
        for i in range(len(Level4UniqueB)):
            Level4.append(self.CanonicalToGrid(int(Level4UniqueB[i])))
            FromBottom.append(Level4UniqueB[i])
        
        # Level 5
        Level5 = []
        Level5Canonical = []
        for i in range(len(Level4)):
            for j in range(n-1):
                if j == 2:
                   j = 3
                grid = self.move_top_half(Level4[i],2-j)
                Level5.append(grid)
                grid = self.move_bottom_half(Level4[i],2-j)
                Level5.append(grid)
                grid = self.move_left_half(Level4[i],2-j)
                Level5.append(grid)
                grid = self.move_right_half(Level4[i],2-j)
                Level5.append(grid)
        
        for i in range(len(Level5)):
            Level5Canonical.append(self.findcanonical(Level5[i]))
        
        Level5UniqueB, indecies5 = np.unique(Level5Canonical, return_index = True)
        Level5 = []
        for i in range(len(Level5UniqueB)):
            Level5.append(self.CanonicalToGrid(int(Level5UniqueB[i])))
            FromBottom.append(Level5UniqueB[i])
            
        # Level 6
        Level6 = []
        Level6Canonical = []
        for i in range(len(Level5)):
            for j in range(n-1):
                if j == 2:
                   j = 3
                grid = self.move_top_half(Level5[i],2-j)
                Level6.append(grid)
                grid = self.move_bottom_half(Level5[i],2-j)
                Level6.append(grid)
                grid = self.move_left_half(Level5[i],2-j)
                Level6.append(grid)
                grid = self.move_right_half(Level5[i],2-j)
                Level6.append(grid)
        
        for i in range(len(Level6)):
            Level6Canonical.append(self.findcanonical(Level6[i]))
        
        Level6UniqueB, indecies6 = np.unique(Level6Canonical, return_index = True)
        Level6 = []
        for i in range(len(Level6UniqueB)):
            Level6.append(self.CanonicalToGrid(int(Level6UniqueB[i])))
            FromBottom.append(Level6UniqueB[i])
        
        self.Level6UniqueB = Level6UniqueB
        self.Level5UniqueB = Level5UniqueB
        self.Level4UniqueB = Level4UniqueB
        self.Level3UniqueB = Level3UniqueB
        self.Level2UniqueB = Level2UniqueB
        self.Level1UniqueB = Level1CanonicalB
        
   
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
        #print(self.state)
        self.state = self.bitstring_to_grid(self.state)
        #print('State after transformed to grid and before step:')
        #print(self.state)
        #print('')
        stateM = self.state
        slide = action
            #First its gonna do the action that was chosen by the agent.
        if slide == 0:
            stateM = self.move_left_half(stateM, 1)
        elif slide == 1:
            stateM = self.move_left_half(stateM, -1)
        elif slide == 2:
            stateM = self.move_left_half(stateM, 2)
        elif slide == 3:
            stateM = self.move_right_half(stateM, 1)
        elif slide == 4:
            stateM = self.move_right_half(stateM, -1)
        elif slide == 5:
            stateM = self.move_right_half(stateM, 2)
        elif slide == 6:
            stateM = self.move_top_half(stateM, -1)
        elif slide == 7:
            stateM = self.move_top_half(stateM, 1)
        elif slide == 8:
            stateM = self.move_top_half(stateM, 2)
        elif slide == 9:
            stateM = self.move_bottom_half(stateM, -1)
        elif slide == 10:
            stateM = self.move_bottom_half(stateM, 1)
        elif slide == 11:
            stateM = self.move_bottom_half(stateM, 2)
        else:
            raise ValueError("Unrecognized direction: {}".format(slide))

        #print('After Step is completed, state is:')
        #print(self.state)
        canonical = self.findcanonical(stateM)
            #checks list of past actions to check if it has been there before
#        iso = 0
#        for i in range(len(self.stateList)):
#            if self.stateList[i] == canonical:
#                iso += 1
                #print('Found iso')
        #print('')
        #print('Canonical is:')
        #print(canonical)
        
                
#        if iso >= 1:
#            stateM = self.random_action()
            #print(stateM)
#            canonical = self.findcanonical(stateM)
            #print('New canonical is:')
            #print(canonical)
#            self.stateList.append(canonical)
#        else:
#            self.stateList.append(canonical)
        #print('')
        #print('List of past actions:')
        #print(self.stateList)
        
        
        #print(self.stateList)
#        #check if done:
        if canonical == self.doneState:
            reward = 1
            done = True
            #print(self.stateList)
        else:
            reward = 0
            done = False
        #print('what')
        #print(self.state)
        self.state = self.CanonicalToGrid(canonical)
#        if (self.state == 2).sum() > 4:
#            np.place(self.state, self.state==1, 5)
#            np.place(self.state, self.state==2, 1)
#            np.place(self.state, self.state==5, 2)
#        if (self.state == 3).sum() > 4:
#            np.place(self.state, self.state==1, 5)
#            np.place(self.state, self.state==3, 1)
#            np.place(self.state, self.state==5, 3)
        state1 = np.zeros(16)
        for i in range(4):
            for j in range(4):
                state1[i*4+j] = self.state[i,j]
        bits = self.convert_to_bit(state1)
        self.steps_beyond_done = None
        self.state = state1
        #print('')
        #print('State at the end of the step function:')
        #print(self.state)
            
        return self.state,  reward, done, {}   
            
#reset    
    def reset(self):
        #state = self.generate_two(4)
        #state = self.generate_threeColour(4)
        state = self.generate_threeColour_initialise(4)
        #state = self.generate_fourColour(4)
        #state = self.generate_fourColour_initialise(4)
        state = state.astype(int)
        state1 = np.zeros(16)
        for i in range(4):
            for j in range(4):
                state1[i*4+j] = state[i,j]
        bits = self.convert_to_bit(state1)
        self.steps_beyond_done = None
        self.stateList = []
        self.state = bits.bin
        self.state = state1
        #print(self.state)
        return self.state
        #set to a random state

 
    #not going to render anything
    #def render(self, mode = None):
    
    '''
    other functions:
    '''
#    def move(self, action, grid):
#        self.DIRECTIONS = ['lhu', 'lhd', 'rhu', 'rhd', 'thl', 'thr', 'bhl', 'bhr']
#        
#        stateM = grid
#        iso = 0
#        rand = 0
#        while iso == 0:
#            '''
#            probleem is dat hy kan hierso vashaak en vreeslik baie steps doen net om by
#            'n nuwe state uit te kom.
#            
#            So, voorheen het ek net gecheck of canonical op lys is na agent se action, en as dit was
#            enige random action gekies (al het dit result in a canonical wat al op die lys is of die
#            random action die agent se action is {letterlik net random action = random.randrange(8)}) 
#            en aanbeweeg. 
#            
#            My probleem nou is dat as ek 'n random action moet kies totdat ons by 'n state uitkom wat
#            nie equivalent is aan 'n vorige state nie, dan kan dit veroorsaak dat dit in 'n loop hier vashaak
#            en tegnies meer as een move/step doen. Gaan dit nie die algorithm deurmekaar maak as die
#            agent, byvoorbeeld, action 2 kies, maar dan in die agtergrond word daar 100 steps gedoen
#            want alles is equivalent aan vorige states. Dan kry die agent 'n observation wat hy dink
#            is as gevolg van action 2, maar is eintlik 'n kombinasie van 100 actions? Want die agent
#            is eintlik net supposed om een uit 8 actions te kies wat result in 1 'slide' move in een van
#            die 8 rigtings, maar a.g.v. hierdie loop kan daar amper 'n infinite amount of slides gedoen word
#            in verskillende rigtings.
#            
#            en wat as die beste manier om die puzzle te solve is om terug te beweeg na 'n vorige state toe?
#            
#            '''
#            slide = self.DIRECTIONS[action]
#            #First its gonna do the action that was chosen by the agent.
#            if slide == 'lhu':
#                stateM = self.move_left_half(stateM, 1)
#            elif slide == 'lhd':
#                stateM = self.move_left_half(stateM, -1)
#            elif slide == 'rhu':
#                stateM = self.move_right_half(stateM, 1)
#            elif slide == 'rhd':
#                stateM = self.move_right_half(stateM, -1)
#            elif slide == 'thl':
#                stateM = self.move_top_half(stateM, -1)
#            elif slide == 'thr':
#                stateM = self.move_top_half(stateM, 1)
#            elif slide == 'bhl':
#                stateM = self.move_bottom_half(stateM, -1)
#            elif slide == 'bhr':
#                stateM = self.move_bottom_half(stateM, 1)
#            else:
#                raise ValueError("Unrecognized direction: {}".format(slide))
#            #print('after first move')
#            #print(stateM)
#            #get canonical
#            canonical = self.findcanonical(stateM)
#            #checks list of past actions to check if it has been there before
#            for i in range(len(self.stateList)):
#                if self.stateList[i] == canonical:
#                    iso += 1
#                    #print('Found iso')
#                
#            if iso == 0 : #current state has not is not equivalent to past states
#                #print('did not find iso')
#                break
#                
#            else:
#                iso = 0 #so that while loop runs again
#                #print('random action')
#                action = random.randrange(8)
#                rand += 1
##                if rand > 6: #not sure about this, it just breaks out of while loop
##                    break
#            
#        return stateM
##            if iso > 0:
##                #print('')
##                #print('Has been there before, selecting new random action.')
##                self.state = self.random_action()       #if it has been there before, select random action
##                #print('State after random action:')
##                #print(self.state)
##                iso = 0
##                canonical = self.findcanonical(self.state)
##                
##                for i in range(len(self.stateList)):                        #checks list of past actions to check if it has been there before
##                    if self.stateList[i] == canonical:
##                        iso += 1
##                while iso > 0:
            
                    
            

    def bitstring_to_grid(self, bitstring):
        togrid = bitstring[0:]
        state = np.zeros((4, 4))
        k=0
        for i in range(4):
            for j in range(4):
                state[i,j] = togrid[k]
                k=k+1
#        for i in range(0, 4):
#            j=0
#            for j in range(0, 4):
#                if togrid[k * 2:k * 2 + 2] == '00':
#                    state[i][j] = 1
#                    # print(k*2,k*2+2)
#                    # print(i,j)
#                elif togrid[k * 2:k * 2 + 2] == '01':
#                    state[i][j] = 2
#                    # print(k * 2, k * 2 + 2)
#                    # print(i,j)
#                elif togrid[k * 2:k * 2 + 2] == '10':
#                    state[i][j] = 3
#                    # print(k * 2, k * 2 + 2)
#                    # print(i,j)
#                elif togrid[k * 2: k * 2 + 2] == '11':
#                    state[i][j] = 4
#                    # print(k * 2, k * 2 + 2)
#                    # print(i,j)
#                k += 1
#                j += 1
#            i += 1
    
        return state.astype(int)
    
    def random_action(self):

        #self.DIRECTIONS = ['lhu', 'lhd', 'rhu', 'rhd', 'thl', 'thr', 'bhl', 'bhr']
        rand_action = random.randrange(12)
        #print('with random number of:')
        #print(rand_action)
        slide = rand_action
  #      self.state = self.bitstring_to_grid(self.state)
        stateM = self.state
        if slide == 0:
            stateM = self.move_left_half(stateM, 1)
        elif slide == 1:
            stateM = self.move_left_half(stateM, -1)
        elif slide == 2:
            stateM = self.move_left_half(stateM, 2)
        elif slide == 3:
            stateM = self.move_right_half(stateM, 1)
        elif slide == 4:
            stateM = self.move_right_half(stateM, -1)
        elif slide == 5:
            stateM = self.move_right_half(stateM, 2)
        elif slide == 6:
            stateM = self.move_top_half(stateM, -1)
        elif slide == 7:
            stateM = self.move_top_half(stateM, 1)
        elif slide == 8:
            stateM = self.move_top_half(stateM, 2)
        elif slide == 9:
            stateM = self.move_bottom_half(stateM, -1)
        elif slide == 10:
            stateM = self.move_bottom_half(stateM, 1)
        elif slide == 11:
            stateM = self.move_bottom_half(stateM, 2)
        else:
            raise ValueError("Unrecognized direction: {}".format(slide))
        return stateM
        

    #@staticmethod
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
                    #print(rand)
                    if (rand==1 and (one<int(((n/2)**2)*2))):
                            initial[i][j] = rand
                            one=one+1
    
                    # elif one==int((n/2)**2):
                    #     while rand==1:
                    #         rand = random.randrange(n) + 1
    
                    if (rand == 2 and (two < int(((n/2)**2)))):
                            initial[i][j] = rand
                            two = two + 1
                    # elif two==int((n/2)**2):
                    #
                    #     while rand == 2:
                    #         rand = random.randrange(n) + 1
    
                    if (rand == 3 and (three < int(((n / 2) ** 2) ))):
                            initial[i][j] = rand
                            three = three + 1
    
                    # elif three==int((n/2)**2):
                    #     while rand == 3:
                    #         rand = random.randrange(n) + 1
    
    
    
                    # elif four==int((n/2)**2):
                    #     while rand == 4:
                    #         rand = random.randrange(n) + 1
    
                    #print("i",i,"j",j,initial[i][j])
        return initial
    
    def generate_threeColour_initialise(self, n):
        rand = random.randint(0,len(self.Level3UniqueB)-1)
        canonical = self.Level3UniqueB[rand]
        initial = self.CanonicalToGrid(canonical)
#        if (initial == 2).sum() > 4:
#            np.place(initial, initial==1, 5)
#            np.place(initial, initial==2, 1)
#            np.place(initial, initial==5, 2)
#        if (initial == 3).sum() > 4:
#            np.place(initial, initial==1, 5)
#            np.place(initial, initial==3, 1)
#            np.place(initial, initial==5, 3)
#        print(initial)
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
                    #print(rand)
                    if (rand==1 and (one<int((n/2)**2))):
                            initial[i][j] = rand
                            one=one+1
    
                    # elif one==int((n/2)**2):
                    #     while rand==1:
                    #         rand = random.randrange(n) + 1
    
                    if (rand == 2 and (two < int(((n/2)**2)))):
                            initial[i][j] = rand
                            two = two + 1
                    # elif two==int((n/2)**2):
                    #
                    #     while rand == 2:
                    #         rand = random.randrange(n) + 1
    
                    if (rand == 3 and (three < int(((n / 2) ** 2) ))):
                            initial[i][j] = rand
                            three = three + 1
    
                    # elif three==int((n/2)**2):
                    #     while rand == 3:
                    #         rand = random.randrange(n) + 1
    
                    if (rand == 4 and (four < int((n / 2) ** 2 ))):
                            initial[i][j] = rand
                            four = four + 1
    
                    # elif four==int((n/2)**2):
                    #     while rand == 4:
                    #         rand = random.randrange(n) + 1
    
                    #print("i",i,"j",j,initial[i][j])
        return initial
    
    def generate_fourColour_initialise(self, n):
        rand = random.randint(0,len(self.Level6UniqueB)-1)
        canonical = self.Level6UniqueB[rand]
        initial = self.CanonicalToGrid(canonical)
 #       if (initial == 2).sum() > 4:
 #           np.place(initial, initial==1, 5)
 #           np.place(initial, initial==2, 1)
 #           np.place(initial, initial==5, 2)
 #       if (initial == 3).sum() > 4:
 #           np.place(initial, initial==1, 5)
 #           np.place(initial, initial==3, 1)
 #           np.place(initial, initial==5, 3)
#        print(initial)
        return initial

#Convert array of state (grid) into a bitstream
    def convert_to_bit(self, state_array):
        bits = BitStream(bin='00000000000000000000000000000000')
        for i in range(0, 16):
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
        gridnew[0] = np.roll(gridnew[0], n)
        gridnew[1] = np.roll(gridnew[1], n)
        return gridnew
    
    def move_bottom_half(self, grid, n):
        gridnew = copy.deepcopy(grid)
        gridnew[2] = np.roll(gridnew[2], n)
        gridnew[3] = np.roll(gridnew[3], n)
        return gridnew
    
    def move_left_half(self, grid, n):
        gridtemp = copy.deepcopy(grid)
        gridnew = np.transpose(gridtemp)
        gridnew[0] = np.roll(gridnew[0], n)
        gridnew[1] = np.roll(gridnew[1], n)
        return np.transpose(gridnew)
    
    def move_right_half(self, grid, n):
        gridtemp = copy.deepcopy(grid)
        gridnew = np.transpose(gridtemp)
        gridnew[2] = np.roll(gridnew[2], n)
        gridnew[3] = np.roll(gridnew[3], n)
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
        n = 2
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
    
        for t in range(0, 4):
            for v in range(0, 4):
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
        j = 0
        for j in range(0, 32):
            i = 0
            bits = BitStream(bin='00000000000000000000000000000000')
            for i in range(0, 16):
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
    
        # print("Original")
        # print(iso1)
        # print("Rotation 1")
        # print(iso2)
        # print("Rotation 2")
        # print(iso3)
        # print("Rotation 3")
        # print(iso4)
    
    
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
    
        # maak 'n loop wat toets of al 'n 1 2 3 en 4 teegekom het sodat nie elke keer als doen nie. Andersins gaan 13 keer
        #  moet hardloop
    
        colours = self.threeD_to_lists(isomorphs)
    
        canonical = min(self.convert_to_smallestint(colours)) #kan verander dat dit 'n 32 bit string return
    
        #print(canonical)
        return canonical
    
    def CanonicalToGrid(self, canonical):
        togrid = bin(canonical)[2:].zfill(32)
        state = np.zeros((4, 4))
        k=0
        for i in range(0, 4):
            j=0
            for j in range(0, 4):
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
    
        