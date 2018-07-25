import os
import torch
import random
from collections import namedtuple

## REPLAY MEMORY
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','ended'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class CheckpointIfBetter:
    def __init__(self, param, device, w = 0.95):
        self.w = w
        self.param = param
        self.n = 0
        self.run_loss = 0
        self.best_loss = 0
        self.device = device
        
        if not os.path.isdir("checkpoints/"):
            os.makedirs("checkpoints/")
        
    def save(self, fullmodel, step, step_loss):
        if self.n < 30:
            self.run_loss = (self.run_loss*self.n + step_loss)/(self.n+1)
        else:
            self.run_loss = self.run_loss*self.w + step_loss*(1-self.w)
            
        if self.best_loss > self.run_loss:
            filename = "checkpoints/%s-step:%s-loss:%.2f.pth" %( self.param['version'], step, self.run_loss)
            torch.save(fullmodel.cpu().state_dict(), filename)
            fullmodel.to(self.device)
            self.best_loss = self.run_loss
            
        self.n +=1
        return True
    
class EpsilonDecay():
    def __init__(self, start_eps = 0.99, end_eps = 0.05, length = 1000000):
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.decay = (end_eps-start_eps)/length
        
    def get(self, step):
        return max(self.start_eps + self.decay*step, self.end_eps)