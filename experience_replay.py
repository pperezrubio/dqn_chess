import numpy as np
from collections import deque

class Memory(object):
    """
    Represents a (state_t, action_t, reward_(t+1),state_(t+1)) 
    sequence observed. Can be 
    """
    def __init__(self, st, at, rt, stp1):
        """
        st - state matrix from time t. 
        at - action chosen at time t.
        rt - scalar reward observed at time t+1.
        stp1 - state matrix from time t+1. 
        """
        self.st = st
        self.at = at
        self.rt = rt
        self.stp1 = stp1

    def target_pair(self, nn):
        # compute an (input, target) pair with respect to a neural network. 
        # parametrizing the action value function. 
        current_out = nn.output(st)
        nxt_out = nn.output(stp1)
        target = current_out.copy()
        # target is only different for the action chosen. 
        target[at] = np.max(nn.output(stp1)) + rt
        return (st, target)

class ExperienceReplay(object):
    """
    Acts as a repository of past experiences. Should be queue-like,
    in that it has a fixed size, and the first memories discarded are those
    which we experienced earliest. 
    """
    def __init__(self, max_size):
        self.size = 0
        self.max_size = max_size
        self.d = deque()

    def pop(self):
        try:
            self.d.popleft()
            self.size -= 1
        except IndexError:
            raise RuntimeError("Can't pop from an empty queue.")
        
    def insert(self, memory):
        if self.size == self.max_size:
            self.pop()
        self.d.append(memory)
        self.size += 1
        
    def sample(self, N):
        return np.random.choice(self.d, N, replace=True)

