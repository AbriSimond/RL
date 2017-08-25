import multiprocessing as mp
import gym
import numpy as np
from joblib import Parallel, delayed
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
        
    def play_game(self,env=None):
        if env is None:
            env = gym.make("Pong-v0")
            
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
        print('Game done.')
        return {'obs' : obs_hist,'action': action_hist, 'reward' : reward_hist}
        
    def play_multiple_games(self):
        return Parallel(n_jobs=self.workers)(delayed(self.play_game)() for g in range(self.replays))