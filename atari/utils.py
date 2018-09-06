import os
import torch
import random
from collections import namedtuple
from wrappers import wrap_deepmind
import gym

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
    
    
## GAMEPLAY
def play_game(env = wrap_deepmind(gym.make("Pong-v0"), frame_stack = True), agent = None, skipframe = 4, th = 0, maxstep = 5000, render = False, memory = ReplayMemory(50000)):
    cum_reward = 0.0
    render_frames = []
    state = env.reset()
    

    for i in range(maxstep):
        # take action:
        action = agent(state, th = th)
        reward = 0
        for _ in range(skipframe):
            next_state, r, ended, info = env.step(action)
            reward += r
            if ended:
                break
        
        cum_reward += float(reward)
        
        # push to replay buffer:
        memory.push(state, action, next_state, reward, ended)
        state = next_state
        
        if render:
            if i % 1 == 0:
                render_frames.append(torch.from_numpy(env.render(mode="rgb_array")).unsqueeze(0))
        if ended == 1:
            break
            
    out = {'cum_reward' : cum_reward, 'steps' :  i}
    if render:
        out['frames'] = torch.cat(render_frames).permute(3,0,1,2).unsqueeze(0)
    return out
