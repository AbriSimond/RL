import numpy as np
import datetime

def discount_rewards(r,gamma=0.99):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def concat_games(games):
    '''send in list of results from playgame in PlayGym'''
    allgames = {}
    for k in games[0].keys():
        store = [g[k] for g in games]
        allgames[k] = np.concatenate(store,0)
    return allgames


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from keras.models import load_model

class Karpathy_Agent:
    def __init__(self,model_path=None):
        self.prev_state = np.zeros((1,6400))
        self.run_reward = -20.5
        self.epsilon = 0.1
        if model_path == None:
            self.make_model()
        else:
            print("Loading model from checkpoint %s" % model_path)
            self.model = load_model(model_path)
    def make_model(self):
        model = Sequential()
        model.add(Dense(units=200,input_shape=(6400,), activation="relu"))
        model.add(Dense(6, activation="softmax"))
        model.compile(loss='sparse_categorical_crossentropy', 
                       optimizer=RMSprop(lr=0.01))
        self.model = model
        
    def predict(self,state):
        if np.random.random() > self.epsilon:
            proc_state = prepro(state).reshape(1,-1)
            x, self.prev_state = proc_state - self.prev_state, proc_state
            pred = self.model.predict(x).flatten()
            return np.random.choice(np.arange(6),p=pred) # hardcoded
        else:
            return np.random.choice(np.arange(6)) # hardcoded
        
    def process_one_game(self,onegame):
        discounted_rewards = discount_rewards(onegame['reward'])

        
        obs = onegame['obs']
        obs_proc = np.zeros((obs.shape[0],6400))
        for i, frame in enumerate(obs):
            obs_proc[i,:] = prepro(frame)
        
        # Take difference
        obs_diff = obs_proc-np.roll(obs_proc,1,axis=0)
        obs_diff[0,] = 0
        
        return {'obs' : obs_diff, 
                'reward' : onegame['reward'], 
                'reward_discounted' : discounted_rewards,
                'action' : onegame['action']}
    
    def fit_games(self,game_result):
        dat = concat_games(game_result)
        
        dat['reward_discounted'] -= np.mean(dat['reward_discounted'])
        dat['reward_discounted'] /= np.std(dat['reward_discounted'])
        self.model.train_on_batch(dat['obs'],
                                  dat['action'],
                                  sample_weight=dat['reward_discounted'])
        
        # Progress report
        batch_reward = np.sum(dat['reward'])/self.replays
        self.run_reward = self.run_reward*0.95 + 0.05*batch_reward
        print("Batch Reward: %.2f \t Running: %.3f" % (batch_reward,self.run_reward))
        
        # Save if lucky
        if (np.random.random() < 0.1) & (batch_reward > self.run_reward):
            filepath = 'checkpoint/%s_runReward:%.2f_.hdf5' % (datetime.datetime.now(),self.run_reward)
            print('Saving..')
            self.model.save(filepath)
        