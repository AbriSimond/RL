from pygame.locals import *
from random import randint
import pygame
import time
import numpy as np
import datetime
import agents

### WALL ###
class Wall:
    def __init__(self,game_size,box_size):
        self.box_size = box_size
        self.game_size = game_size
        self.X, self.Y = self.make_wall(game_size[0]-1, game_size[1]-1)

    def make_wall(self, x_size,y_size):
        x = np.concatenate([np.array(range(x_size)),np.array(range(x_size)),
                            np.repeat(0,y_size), np.repeat(x_size,y_size)])
        y = np.concatenate([np.repeat(0,y_size), np.repeat(y_size, x_size),
                            np.array(range(y_size)), np.array(range(y_size))])
        return x, y

    def draw(self, surface, image):
        for x, y in zip(self.X,self.Y):
            surface.blit(image,(x*self.box_size, y*self.box_size))

class Apple:
    x = 0
    y = 0

    def __init__(self,x,y, game_size, box_size):
        self.x = x
        self.y = y
        self.game_size = game_size
        self.box_size = box_size
        self.board = np.zeros(game_size)

    def draw(self, surface, image):
        surface.blit(image,(self.x*self.box_size, self.y*self.box_size))

### PLAYER ###
class Player:
    direction = 0
    length = 2

    def __init__(self, length, game_size, box_size):
        self.x = [3,2,1]
        self.y = [2,2,2]
        self.length = length
        self.game_size = game_size
        self.box_size = box_size
        for i in range(0,2000):
            self.x.append(-1)
            self.y.append(-1)

    def update(self):
        # update previous positions
        for i in range(self.length-1,0,-1):
            self.x[i] = self.x[i-1]
            self.y[i] = self.y[i-1]

        # update position of head of snake
        if self.direction == 0:
            self.x[0] = self.x[0] + 1
        if self.direction == 1:
            self.x[0] = self.x[0] - 1
        if self.direction == 2:
            self.y[0] = self.y[0] - 1
        if self.direction == 3:
            self.y[0] = self.y[0] + 1

    def moveRight(self):
        self.direction = 0

    def moveLeft(self):
        self.direction = 1

    def moveUp(self):
        self.direction = 2

    def moveDown(self):
        self.direction = 3

    def draw(self, surface, image):
        for i in range(0,self.length):
            surface.blit(image,(self.x[i]*self.box_size,self.y[i]*self.box_size))

class Snake:
    # Render parameters
    windowHeight = 500
    windowWidth = 500
    box_size = 20

    def __init__(self, render = True, game_size = (10,10), time_reward = -0.02):
        self.game_size = game_size
        self.render = render
        self.time_reward = time_reward
        
        pygame.init()
        if render:
            self._display_surf = pygame.display.set_mode((self.windowWidth,self.windowHeight), pygame.HWSURFACE)
            pygame.display.set_caption('snake game to train agents in')

        # Rendering stuff
        apple_img = 130*np.ones((self.box_size-1,self.box_size-1))
        wall_img = 50* np.ones((self.box_size-1,self.box_size-1))
        snake_img = 200* np.ones((self.box_size-1,self.box_size-1))
        self._image_surf = pygame.surfarray.make_surface(snake_img)
        self._wall_surf = pygame.surfarray.make_surface(wall_img)
        self._apple_surf = pygame.surfarray.make_surface(apple_img)

    def isCollision(self,x1,y1,x2,y2):
        if x1 == x2 and y1 == y2:
            return True
        return False

    def on_init(self):
        self.player = Player(3, self.game_size, self.box_size)
        self.apple = Apple(3,3, self.game_size, self.box_size)
        self.wall = Wall(self.game_size, self.box_size)

        self._running = True
        self.reward = 0
        self.ended = False

    def on_feedback(self):
        state = np.zeros((3, self.game_size[0],self.game_size[1]))

        k = 0 # PLAYER
        for i in range(self.player.length):
            if not self.player.y[i]<0:
                state[k, self.player.y[i], self.player.x[i]] = ( self.player.length - i)/self.player.length

        k = 1 # APPLE
        state[k, self.apple.y, self.apple.x] = 1

        k = 2 # WALLS
        for x, y in zip(self.wall.X,self.wall.Y):
            state[k, y, x] = 1

        return state.round(2), self.reward, self.ended

    def on_render(self):
        self._display_surf.fill((0,0,0))
        self.player.draw(self._display_surf, self._image_surf)
        self.apple.draw(self._display_surf, self._apple_surf)
        self.wall.draw(self._display_surf, self._wall_surf)
        pygame.display.flip()

    def on_loop(self):
        self.player.update()
        self.reward = self.time_reward
        # does snake collide with wall?
        for x, y in zip(self.wall.X, self.wall.Y):
            if self.isCollision(x,y,self.player.x[0], self.player.y[0]):
                #print("You lose! Wall collision. Length: %d" % self.player.length)
                self.reward = -1.0
                self.ended = True


        # does snake collide with itself?
        for i in range(2,self.player.length):
            if self.isCollision(self.player.x[0],self.player.y[0],self.player.x[i], self.player.y[i]):
                #print("You lose! Self-collision. Length: %d" % self.player.length)
                self.reward = -1.0
                self.ended = True

        # does snake eat apple?
        for i in range(0,self.player.length):
            if self.isCollision(self.apple.x,self.apple.y,self.player.x[i], self.player.y[i]):
                self.player.length = self.player.length + 1
                self.apple.x = randint(1,self.game_size[0]-2)
                self.apple.y = randint(1,self.game_size[1]-2)
                self.reward = 1.0

    def step(self, action):
        if action=="right":
            self.player.moveRight()
        elif action=="left":
            self.player.moveLeft()
        elif action=="up":
            self.player.moveUp()
        elif action=="down":
            self.player.moveDown()
        elif action=="escape":
            self._running = False

        self.on_loop()

        if self.render:
            pygame.event.pump()
            self.on_render()

        feedback = self.on_feedback()
        if self.ended:
            self.on_init()

        return feedback

if __name__ == "__main__" :
    snake = Snake(render=True)
    snake.on_init()
    state, reward, ended = snake.on_feedback()
    while True:
        action = agents.simple_agent(state)
        state, reward, ended = snake.step(action)
