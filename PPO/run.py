import PlayGame#File as PlayGame
import karpathy_agent as agent

from importlib import reload
reload(agent)
reload(PlayGame)

games = PlayGame.PlayGym("Pong-v0",workers=1,replays=10)
games.agent = agent.Karpathy_Agent()
games.agent.replays = games.replays

import gym
env = gym.make("Pong-v0")
while True:
    game_result = games.play_multiple_games(env)
    games.agent.fit_games(game_result)

