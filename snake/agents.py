import numpy as np
from pygame.locals import *
import pygame

def max_idx(x):
    row, col = np.where(x == np.max(x))
    return int(row), int(col)

def simple_agent(state):
    y_player, x_player = max_idx(state[0,:,:])
    y_apple, x_apple = max_idx(state[1,:,:])

    if y_player < y_apple:
        return "down"
    elif y_player > y_apple:
        return "up"
    elif x_player < x_apple:
        return "right"
    elif x_player > x_apple:
        return "left"
    else:
        return "up"

def keyboard_agent(state=None):
    keys = pygame.key.get_pressed()
    if (keys[K_RIGHT]):
        return "right"

    elif (keys[K_LEFT]):
        return "left"

    elif (keys[K_UP]):
        return "up"

    elif (keys[K_DOWN]):
        return "down"

    elif (keys[K_ESCAPE]):
        return "escape"

def random_legal_move(state):
    y, x = max_idx(state[0,:,:])
    legal_moves = ((state[0,] + state[2,]) == 0)
    
    test_coord = {
        'left' : (y,x-1),
        'right' : (y,x+1),
        'up' : (y-1, x),
        'down' : (y+1, x)}

    sample_space = [move for move, (y,x) in test_coord.items() if legal_moves[y,x]]
    
    if len(sample_space) > 0:
        return np.random.choice(sample_space)
    else:
        return 'down'