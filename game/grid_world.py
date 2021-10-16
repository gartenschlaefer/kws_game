"""
building a simple grid world
"""

import pygame
import numpy as np

from wall import Wall, MovableWall

from interactable import Interactable


class GridWorld(Interactable):
  """
  grid world class
  """

  def __init__(self, surf, screen_size, color_bag, pixel_size=(20, 20), grid_move=True):
    """
    create the grid world
    """

    # arguments
    self.surf = surf
    self.screen_size = np.array(screen_size)
    self.color_bag = color_bag
    self.pixel_size = np.array(pixel_size)
    self.grid_move = grid_move
    
    # sprites
    self.sprites = pygame.sprite.Group()

    # pixel spacing
    self.grid_size = self.screen_size // self.pixel_size

    # create empty wall grid
    self.wall_grid = np.zeros(self.grid_size)
    self.wall_sprites = pygame.sprite.Group()

    # create empty movable wall grid
    self.move_wall_grid = np.zeros(self.grid_size)
    self.move_wall_sprites = pygame.sprite.Group()

    # move wall container
    self.move_walls = []

    # active move wall
    self.act_wall = 0


  def grid_to_pos(self, grid_pos):
    """
    transform grid to position
    """

    return (grid_pos[0] * self.pixel_size[0], grid_pos[1] * self.pixel_size[1])


  def create_walls(self):
    """
    create walls
    """

    # normal walls
    for i, wall_row in enumerate(self.wall_grid):
      for j, wall in enumerate(wall_row):

        # normal wall found
        if wall:

          # create wall element at pixel position
          wall = Wall(position=np.array([i, j])*self.pixel_size, color=self.color_bag.wall, size=self.pixel_size)

          # add to sprite groups
          self.wall_sprites.add(wall)

    # movable walls
    for i, move_wall_row in enumerate(self.move_wall_grid):
      for j, move_wall in enumerate(move_wall_row):

        # movable wall found
        if move_wall:

          # create wall element at pixel position
          move_wall = MovableWall(grid_pos=[i, j], color=self.color_bag.default_move_wall, size=self.pixel_size, grid_move=self.grid_move)

          # set grid
          move_wall.set_move_wall_grid(self.move_wall_grid)

          # wall container
          self.move_walls.append(move_wall)

          # add to sprite groups
          self.move_wall_sprites.add(move_wall)

    # init move walls
    self.move_walls_init()

    # add to sprites
    self.sprites.add(self.wall_sprites, self.move_wall_sprites)


  def move_walls_init(self):
    """
    init move walls
    """

    # collision grouping for movable walls
    for i, move_wall in enumerate(self.move_walls):

      # reset moving walls
      move_wall.reset()

      # deactivate move wall
      move_wall.is_active = False
      move_wall.set_color(self.color_bag.default_move_wall)

      # move able wall sees wall
      move_wall.obstacle_sprites.add(self.wall_sprites)

      # sees also moving walls
      sp = self.move_walls.copy()
      sp.pop(i)
      move_wall.obstacle_sprites.add(sp)

    # set one move wall active
    if self.move_walls:

      # set active
      self.move_walls[self.act_wall].is_active = True
      self.move_walls[self.act_wall].set_color(self.color_bag.active_move_wall)


  def action_key(self):
    """
    if action key is pressed
    """

    if not len(self.move_walls): return

    # old wall
    self.move_walls[self.act_wall].is_active = False
    self.move_walls[self.act_wall].set_color(self.color_bag.default_move_wall)

    # index wall
    self.act_wall = self.act_wall + 1 if self.act_wall < len(self.move_walls) - 1 else 0

    # new wall
    self.move_walls[self.act_wall].is_active = True
    self.move_walls[self.act_wall].set_color(self.color_bag.active_move_wall)


  def reset(self):
    """
    reset grid world
    """

    # active move wall
    self.act_wall = 0

    # init move walls again
    self.move_walls_init()


  def is_moveable(self):
    """
    moveable flag
    """
    return True


  def speech_command(self, command):
    """
    move character to position
    """
    self.action_key() if command == 'go' else [move_wall.speech_command(command) for move_wall in self.move_walls if move_wall.is_active]


  def direction_change(self, direction):
    """
    move character to position
    """
    [move_wall.direction_change(direction) for move_wall in self.move_walls if move_wall.is_active]


  def update(self):
    """
    frame update
    """
    [move_wall.update() for move_wall in self.move_walls]


  def draw(self):
    """
    draw
    """
    self.sprites.draw(self.surf)



if __name__ == '__main__':
  """
  Main Gridworld
  """

  import yaml

  # append paths
  import sys
  sys.path.append("../")

  from classifier import Classifier
  from mic import Mic
  from levels import LevelMoveWalls
  from game_logic import GameLogic


  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))


  # --
  # mic

  # create classifier
  classifier = Classifier(cfg_classifier=cfg['classifier'], root_path='../')

  # create mic instance
  mic = Mic(classifier=classifier, mic_params=cfg['mic_params'], is_audio_record=False)


  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(cfg['game']['screen_size'])


  # --
  # level

  # level setup
  level = LevelMoveWalls(screen, cfg['game']['screen_size'], mic)

  # add clock
  clock = pygame.time.Clock()

  # init stream
  mic.init_stream()

  # mic stream and update
  with mic.stream:

    # game loop
    while level.runs():
      for event in pygame.event.get():

        # event handling
        level.event_update(event)

      # frame update
      level.update()

      # update display
      pygame.display.flip()

      # reduce framerate
      clock.tick(cfg['game']['fps'])

  # end pygame
  pygame.quit()
