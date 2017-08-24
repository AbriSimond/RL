import multiprocessing
import numpy as np
import gym

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
    def __init__(self,gamename):
        self.gamename = gamename
        self.env = gym.make(gamename)
        
    def play_game(self,agent):
        obs = self.env.reset()
        obs_hist = []
        reward_hist = []
        action_hist = []
        while True:
            # Execute
            action = agent.predict(obs)
            obs, reward, done, info = self.env.step(action)

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
        return {'obs_hist' : obs_hist,'action_hist': action_hist, 'reward_hist' : reward_hist}