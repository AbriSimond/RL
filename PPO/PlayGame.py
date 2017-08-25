import multiprocessing as mp
import gym
import numpy as np

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
        self.games = [gym.make(gamename) for i in range(workers)]
        
    def play_game(self,env):
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
        return {'obs_hist' : obs_hist,'action_hist': action_hist, 'reward_hist' : reward_hist}
    def play_games(self,env):
        return [self.play_game(env) for i in range(self.replays)]     
    
    def worker_parallel(self,g,queue):
        game_results = self.play_games(g)
        queue.put(game_results)
        
    def play_parallel_queue(self):
        queue = mp.Queue()
        processes = []
        rets = []
        for i, g in enumerate(self.games):
            p = mp.Process(target=self.worker_parallel, args=(g,queue))
            processes.append(p)
            p.start()
        for p in processes:
            ret = queue.get() # will block
            rets.append(ret)
        for p in processes:
            p.join()
        return rets

    def play_parallel_oldqueue(self):
        jobs = []
        game_results = {}
        queue = mp.Queue()
        queue.put(game_results)

        for i,g in enumerate(self.games):
            p = mp.Process(target=self.worker_parallel, args=(g,queue))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
        return True #game_results.values()

    
    

        
    def play_parallel_manager(self):
        jobs = []
        manager = mp.Manager()
        game_results = manager.dict()

        for i,g in enumerate(self.games):
            p = mp.Process(target=self.worker_parallel, args=(g,queue))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
        return True #game_results.values()