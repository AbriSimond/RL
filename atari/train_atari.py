import gym
import torch
from wrappers import wrap_deepmind
from tensorboardX import SummaryWriter
import datetime
import random
from torch import nn
import torch.nn.functional as F
from importlib import reload 
import model
from torch import optim
import numpy as np
import utils

## GAMEPLAY
def play_game(env = wrap_deepmind(gym.make("Pong-v0"), frame_stack = True), agent = None, skipframe = 4, th = 0, maxstep = 5000, render = False):
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


## AGENTS ##

def random_agent(state, th = None):
    return random.randint(a=0,b=env.action_space.n-1)

def dqn_epsilon_agent(state, th = 0.05):
    if random.random() > th:
        yhat = dqn(default_states_preprocessor(state))
        return int(yhat.argmax().cpu().numpy())
    else:
        return env.action_space.sample()
    
    
## PREPROC AND TRAIN ##
def default_states_preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    
    Obtained from https://github.com/Shmuma/ptan/blob/master/ptan/agent.py
    """
    
    if not isinstance(states,list):
        states = [states]
    
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states).permute(0,3,1,2).float().to(device)/255.


def train_batch(param):
    if len(memory) < param['batch_size']:
        return 0
    batch = memory.sample(param['batch_size'])
    batch_states = default_states_preprocessor([m.state for m in batch])
    batch_next_states = default_states_preprocessor([m.next_state for m in batch])
    batch_ended = torch.tensor([m.ended for m in batch])
    batch_rewards = torch.tensor([m.reward for m in batch]).to(device)
    batch_actions = torch.tensor([m.action for m in batch]).to(device)

    ## Calculate expected reward:
    with torch.set_grad_enabled(False):
        not_ended_batch = 1 -torch.ByteTensor(batch_ended).to(device)
        next_states_non_final = batch_next_states[not_ended_batch]
        next_state_values = torch.zeros(param['batch_size']).to(device)
        reward_hat = dqn(next_states_non_final)
        next_state_values[not_ended_batch] = reward_hat.max(1)[0]
        expected_state_action_values = next_state_values*param['GAMMA'] + batch_rewards

    # Predict value function:
    yhat = dqn(batch_states)
    state_action_values = yhat.gather(1, batch_actions.unsqueeze(1)).squeeze()

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    for param in dqn.parameters():
        param.data.clamp_(-1, 1)
    optimizer.step()
    return float(loss.data.cpu().numpy())

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="increase output verbosity", default = "MsPacman-v0")
    args = parser.parse_args()
    print(args)
    
    param = {'env' : args.env,
             'batch_size' : 32,
             'lr' : 0.0001,
            'GAMMA' : 0.95,
            'replay_buffer' : 50000,
            'exp_length' : 1000000}
    param['version'] = ", ".join([ "{}:{}".format(key,val) for key, val in param.items()]) + " "+str(datetime.datetime.now())[:16]
    print(param['version'])

    memory = utils.ReplayMemory(param['replay_buffer'])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    eps = utils.EpsilonDecay(start_eps = 1.0, end_eps = 0.05, length = param['exp_length'])
    writer = SummaryWriter(log_dir = "tensorboard/" + param['version'])
    checkpoint = utils.CheckpointIfBetter(param, device)

    env = wrap_deepmind(gym.make(param['env']), frame_stack = True)
    dqn = model.DQN(num_actions = env.action_space.n).to(device)
    optimizer = optim.Adam(dqn.parameters(), lr = param['lr'])

    # Warmup buffer
    for _ in range(5):
        game = play_game(env, agent = dqn_epsilon_agent, th = eps.get(0))

    step = 0
    loss, rewards, episode_steps = {}, {}, {}
    episode = 0
    while True:
        episode += 1

        ## PLAY GAME
        game = play_game(env, agent = dqn_epsilon_agent, th = eps.get(step))
        rewards['run_reward'], episode_steps['run_episode_steps'] = game['cum_reward'], game['steps']
        step += episode_steps['run_episode_steps']

        ## TRAIN
        for _ in range(episode_steps['run_episode_steps']//param['batch_size']):
            loss['run_loss'] = train_batch(param)


        # Test agent:
        if episode % 10 == 0:
            game = play_game(env, agent = dqn_epsilon_agent, th = 0.02)
            rewards['test_reward'], episode_steps['test_episode_steps'] = game['cum_reward'], game['steps']
            checkpoint.save(dqn, step = step, step_loss = -rewards['test_reward'])


        # REPORTING
        if episode % 5 == 0:
            writer.add_scalars("loss", tag_scalar_dict=loss, global_step= step)
            writer.add_scalars("rewards", rewards, step)
            writer.add_scalar("episode", episode, global_step = step)
            writer.add_scalar("episode_length", episode_steps['run_episode_steps'], global_step = step)
            writer.add_scalar("epsilon", eps.get(step), global_step = step)


        # Animate agent:
        if episode % 500 == 0:
            print("episode: {}, step: {}, reward: {}".format(episode, step, rewards['run_reward']))
            game = play_game(env, agent = dqn_epsilon_agent, th = 0.02, render = True)
            writer.add_video("test_game", game['frames'], global_step = step)