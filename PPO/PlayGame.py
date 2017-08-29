import gym
import numpy as np
from joblib import Parallel, delayed
import _pickle as cPickle
import os

#%matplotlib inline
import matplotlib.pyplot as plt
def plot_frame(frame):
    return plt.imshow(frame.reshape((80,80)))

# Random agent
class RandomAgent:
    def __init__(self,randint=6):
        self.randint = randint
        return
    def predict(self, state):
        return np.random.randint(self.randint)


class PlayGym:
    def __init__(self,gamename,workers=5,replays=6):
        self.gamename = gamename
        #self.env = gym.make(self.gamename)
        self.replays = replays
        self.workers = workers
        self.worker_replays = round(replays/workers)
        
        if not os.path.exists("tmp"):
            os.makedirs("tmp")
        
    def play_game(self,env=None):
        if env is None:
            env = gym.make(self.gamename)
            
        obs = env.reset()
        agent = self.agent
        obs_hist = []
        reward_hist = []
        action_hist = []
        while True:
            # Execute
            action = agent.predict(obs)
            obs, reward, done, info = env.step(action)

            # Collect variables
            obs_hist.append(obs)
            reward_hist.append(reward)
            action_hist.append(action)
            if done:
                break

        obs_hist = np.array(obs_hist)
        reward_hist = np.array(reward_hist)
        action_hist = np.array(action_hist)
        #print('Game done.')
        full_result = {'obs' : obs_hist,'action': action_hist, 'reward' : reward_hist}
        
        # Process result according to 
        processed_result = self.agent.process_one_game(full_result)
        return processed_result
    
    def play_games(self, env=None):
        if env is None:
            env = gym.make(self.gamename)
        
        game_results = []
        for _ in range(self.worker_replays):
            game_results.append(self.play_game(env))
        return game_results
    
    def play_multiple_games(self):
        results = Parallel(n_jobs=self.workers)(delayed(self.play_games)() for g in range(self.workers))
        return results
            
            
            