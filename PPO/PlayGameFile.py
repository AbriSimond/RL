import gym
import numpy as np
from joblib import Parallel, delayed
import _pickle as cPickle
import os

# Random agent
class RandomAgent:
    def __init__(self,randint=6):
        self.randint = randint
        return
    def predict(self, state):
        return np.random.randint(self.randint)

def concat_games(games):
    '''send in list of results from playgame in PlayGym'''
    allgames = {}
    for k in games[0].keys():
        store = [g[k] for g in games]
        allgames[k] = np.concatenate(store,0)
    return allgames

class PlayGym:
    def __init__(self,gamename,workers=5,replays=6):
        self.gamename = gamename
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
        return {'obs' : obs_hist,'action': action_hist, 'reward' : reward_hist}
    
    def play_games(self, env=None):
        if env is None:
            env = gym.make(self.gamename)
        
        game_results = []
        for _ in range(self.worker_replays):
            game_results.append(self.play_game(env))
            
        filename = 'tmp/'+ str(np.random.rand()) + '.pickle'
        f = open(filename, 'wb')
        cPickle.dump(game_results, f)
        f.close()
        return filename
    
    def play_multiple_games(self):
        outfiles = Parallel(n_jobs=self.workers)(delayed(self.play_games)() for g in range(self.workers))
        results = []
        for filename in outfiles:
            f = open(filename,'rb')
            results.extend(cPickle.load(f))
            f.close()
            os.remove(filename)
        return results