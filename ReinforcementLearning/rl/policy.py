from __future__ import division
import numpy as np

from rl.util import *


class Policy(object):
    """Abstract base class for all implemented policies.

    Each policy helps with selection of action to take on an environment.

    Do not use this abstract base class directly but instead use one of the concrete policies implemented.
    To implement your own policy, you have to implement the following methods:

    - `select_action`

    # Arguments
        agent (rl.core.Agent): Agent used
    """
    def _set_agent(self, agent):
        self.agent = agent

    @property
    def metrics_names(self):
        return []

    @property
    def metrics(self):
        return []

    def select_action(self, **kwargs):
        raise NotImplementedError()

    def get_config(self):
        """Return configuration of the policy

        # Returns
            Configuration as dict
        """
        return {}


class LinearAnnealedPolicy(Policy):
    """Implement the linear annealing policy

    Linear Annealing Policy computes a current threshold value and
    transfers it to an inner policy which chooses the action. The threshold
    value is following a linear function decreasing over time."""
    def __init__(self, inner_policy, attr, value_max, value_min, value_test, nb_steps):
        if not hasattr(inner_policy, attr):
            raise ValueError('Policy does not have attribute "{}".'.format(attr))

        super(LinearAnnealedPolicy, self).__init__()

        self.inner_policy = inner_policy
        self.attr = attr
        self.value_max = value_max
        self.value_min = value_min
        self.value_test = value_test
        self.nb_steps = nb_steps

    def get_current_value(self):
        """Return current annealing value

        # Returns
            Value to use in annealing
        """
        if self.agent.training:
            # Linear annealed: f(x) = ax + b.
            a = -float(self.value_max - self.value_min) / float(self.nb_steps)
            b = float(self.value_max)
            value = max(self.value_min, a * float(self.agent.step) + b)
        else:
            value = self.value_test
        return value

    def select_action(self, **kwargs):
        """Choose an action to perform

        # Returns
            Action to take (int)
        """
        setattr(self.inner_policy, self.attr, self.get_current_value())
        return self.inner_policy.select_action(**kwargs)

    @property
    def metrics_names(self):
        """Return names of metrics

        # Returns
            List of metric names
        """
        return ['mean_{}'.format(self.attr)]

    @property
    def metrics(self):
        """Return metrics values

        # Returns
            List of metric values
        """

        return [getattr(self.inner_policy, self.attr)]

    def get_config(self):
        """Return configurations of LinearAnnealedPolicy

        # Returns
            Dict of config
        """
        config = super(LinearAnnealedPolicy, self).get_config()
        config['attr'] = self.attr
        config['value_max'] = self.value_max
        config['value_min'] = self.value_min
        config['value_test'] = self.value_test
        config['nb_steps'] = self.nb_steps
        config['inner_policy'] = get_object_config(self.inner_policy)
        return config

class SoftmaxPolicy(Policy):
    """ Implement softmax policy for multinimial distribution

    Simple Policy

    - takes action according to the pobability distribution

    """
    def select_action(self, nb_actions, probs):
        """Return the selected action

        # Arguments
            probs (np.ndarray) : Probabilty for each action

        # Returns
            action

        """
        action = np.random.choice(range(nb_actions), p=probs)
        return action

class EpsGreedyQPolicy(Policy):
    """Implement the epsilon greedy policy

    Eps Greedy policy either:

    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """
    def __init__(self, eps=.1):
        super(EpsGreedyQPolicy, self).__init__()
        self.eps = eps

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        """Return configurations of EpsGreedyQPolicy

        # Returns
            Dict of config
        """
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config


class GreedyQPolicy(Policy):
    """Implement the greedy policy

    Greedy policy returns the current best action according to q_values
    """
    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        action = np.argmax(q_values)
        return action


class BoltzmannQPolicy(Policy):
    """Implement the Boltzmann Q Policy

    Boltzmann Q Policy builds a probability law on q values and returns
    an action selected randomly according to this law.
    """
    def __init__(self, tau=1., clip=(-500., 500.)):
        super(BoltzmannQPolicy, self).__init__()
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        return action

    def get_config(self):
        """Return configurations of BoltzmannQPolicy

        # Returns
            Dict of config
        """
        config = super(BoltzmannQPolicy, self).get_config()
        config['tau'] = self.tau
        config['clip'] = self.clip
        return config


class MaxBoltzmannQPolicy(Policy):
    """
    A combination of the eps-greedy and Boltzman q-policy.

    Wiering, M.: Explorations in Efficient Reinforcement Learning.
    PhD thesis, University of Amsterdam, Amsterdam (1999)

    https://pure.uva.nl/ws/files/3153478/8461_UBA003000033.pdf
    """
    def __init__(self, eps=.1, tau=1., clip=(-500., 500.)):
        super(MaxBoltzmannQPolicy, self).__init__()
        self.eps = eps
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        """Return the selected action
        The selected action follows the BoltzmannQPolicy with probability epsilon
        or return the Greedy Policy with probability (1 - epsilon)

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
            probs = exp_values / np.sum(exp_values)
            action = np.random.choice(range(nb_actions), p=probs)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        """Return configurations of MaxBoltzmannQPolicy

        # Returns
            Dict of config
        """
        config = super(MaxBoltzmannQPolicy, self).get_config()
        config['eps'] = self.eps
        config['tau'] = self.tau
        config['clip'] = self.clip
        return config


class BoltzmannGumbelQPolicy(Policy):
    """Implements Boltzmann-Gumbel exploration (BGE) adapted for Q learning
    based on the paper Boltzmann Exploration Done Right
    (https://arxiv.org/pdf/1705.10257.pdf).

    BGE is invariant with respect to the mean of the rewards but not their
    variance. The parameter C, which defaults to 1, can be used to correct for
    this, and should be set to the least upper bound on the standard deviation
    of the rewards.

    BGE is only available for training, not testing. For testing purposes, you
    can achieve approximately the same result as BGE after training for N steps
    on K actions with parameter C by using the BoltzmannQPolicy and setting
    tau = C/sqrt(N/K)."""

    def __init__(self, C=1.0):
        assert C > 0, "BoltzmannGumbelQPolicy C parameter must be > 0, not " + repr(C)
        super(BoltzmannGumbelQPolicy, self).__init__()
        self.C = C
        self.action_counts = None

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        # We can't use BGE during testing, since we don't have access to the
        # action_counts at the end of training.
        assert self.agent.training, "BoltzmannGumbelQPolicy should only be used for training, not testing"

        assert q_values.ndim == 1, q_values.ndim
        q_values = q_values.astype('float64')

        # If we are starting training, we should reset the action_counts.
        # Otherwise, action_counts should already be initialized, since we
        # always do so when we begin training.
        if self.agent.step == 0:
            self.action_counts = np.ones(q_values.shape)
        assert self.action_counts is not None, self.agent.step
        assert self.action_counts.shape == q_values.shape, (self.action_counts.shape, q_values.shape)

        beta = self.C/np.sqrt(self.action_counts)
        Z = np.random.gumbel(size=q_values.shape)

        perturbation = beta * Z
        perturbed_q_values = q_values + perturbation
        action = np.argmax(perturbed_q_values)

        self.action_counts[action] += 1
        return action

    def get_config(self):
        """Return configurations of BoltzmannGumbelQPolicy

        # Returns
            Dict of config
        """
        config = super(BoltzmannGumbelQPolicy, self).get_config()
        config['C'] = self.C
        return config



 
class MyPolicy(Policy):
    """Implements epsilon greedy with exploration selection advantage"""
    
    def __init__(self, eps=0.1):
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
        
        self.eps = eps
    
    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        
        resultants = []
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
        
        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        """Return configurations of EpsGreedyQPolicy

        # Returns
            Dict of config
        """
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config
        
    ###########################################################################
    
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
