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
import copy


## AGENTS ##
def random_agent(state, th = None):
    return random.randint(a=0,b=env.action_space.n-1)

    
    
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
        reward_hat = target_dqn(next_states_non_final)
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
    parser.add_argument("--env", help="increase output verbosity", default = "Pong-v0")
    args = parser.parse_args()
    print(args)
    
    param = {'env' : args.env,
             'batch_size' : 32,
             'lr' : 0.0001,
            'GAMMA' : 0.95,
            'replay_buffer' : 500000,
             'end_eps' : 0.1,
            'exp_length' : 2000000}
    param['version'] = ", ".join([ "{}:{}".format(key,val) for key, val in param.items()]) + " "+str(datetime.datetime.now())[:16]
    print(param['version'])

    memory = utils.ReplayMemory(param['replay_buffer'])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    eps = utils.EpsilonDecay(start_eps = 1.0, end_eps = param['end_eps'], length = param['exp_length'])
    writer = SummaryWriter(log_dir = "tensorboard/" + param['version'])
    checkpoint = utils.CheckpointIfBetter(param, device)

    env = wrap_deepmind(gym.make(param['env']), frame_stack = True)
    dqn = model.DQN(num_actions = env.action_space.n).to(device)
    target_dqn = copy.deepcopy(dqn)
    
    def dqn_epsilon_agent(state, net = dqn, th = 0.05):
        if random.random() > th:
            yhat = net(default_states_preprocessor(state))
            return int(yhat.argmax().cpu().numpy())
        else:
            return env.action_space.sample()

    optimizer = optim.Adam(dqn.parameters(), lr = param['lr'])

    # Warmup buffer
    for _ in range(5):
        game = utils.play_game(env, agent = dqn_epsilon_agent, th = eps.get(0), memory = memory)

    step = 0
    metrics = {}
    metrics['episode'] = 0
    while True:
        metrics['episode'] += 1

        ## PLAY GAME
        metrics['epsilon'] = eps.get(step)
        game = utils.play_game(env, agent = dqn_epsilon_agent, th = metrics['epsilon'], memory = memory)
        metrics['run_reward'], metrics['run_episode_steps'] = game['cum_reward'], game['steps']
        step += metrics['run_episode_steps']

        ## TRAIN
        for _ in range(metrics['run_episode_steps']//param['batch_size']):
            metrics['run_loss'] = train_batch(param)
            
        if metrics['episode'] % 500 == 0:
            target_dqn.load_state_dict(dqn.state_dict())

        # Test agent:
        if metrics['episode'] % 100 == 0:
            game = utils.play_game(env, agent = dqn_epsilon_agent, th = 0.02, memory = memory)
            metrics['test_reward'], metrics['test_episode_steps'] = game['cum_reward'], game['steps']
            checkpoint.save(dqn, step = step, step_loss = -metrics['test_reward'])


        # REPORTING
        if metrics['episode'] % 100 == 0:
            for key, val in metrics.items():
                writer.add_scalar(key, val, global_step = step)
                
        # Animate agent:
        if metrics['episode'] % 2500 == 0:
            print("episode: {}, step: {}, reward: {}".format(metrics['episode'], step, metrics['run_reward']))
            game = utils.play_game(env, agent = dqn_epsilon_agent, th = 0.02, render = True, memory = memory)
            writer.add_video("test_game", game['frames'], global_step = step)