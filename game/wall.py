"""
character class
"""

import pygame
import numpy as np

from input_handler import InputKeyHandler, InputMicHandler
from interactable import Interactable
from moveable import Moveable


class Wall(pygame.sprite.Sprite):
  """
  wall class
  """

  def __init__(self, position, color=(10, 200, 200), size=(20, 20)):

    # MRO check
    super().__init__()

    # vars
    self.position = position
    self.color = color
    self.size = size

    # wall init
    self.image = pygame.surface.Surface(self.size)
    self.rect = self.image.get_rect()

    # set rect position
    self.rect.x = self.position[0]
    self.rect.y = self.position[1]

    self.image.fill(color)


  def set_color(self, color):
    """
    set the color
    """

    self.image.fill(color)



class MovableWall(Wall, Interactable, Moveable):
  """
  a movable wall
  """

  def __init__(self, grid_pos, color=(10, 200, 200), size=(20, 20), grid_move=False):

    # parent init
    super().__init__(np.array(grid_pos)*size, color, size)

    # arguments
    self.grid_pos = grid_pos
    self.grid_move = grid_move

    # moveable init
    Moveable.__init__(self, move_sprite=self, move_rect=self.rect, move_speed=[3, 3], has_gravity=False, grid_move=self.grid_move)
    
    # interactions
    self.obstacle_sprites = pygame.sprite.Group()
    self.is_active = True
    
    # save init pos
    self.init_pos = self.position
    self.init_grid_pos = self.grid_pos.copy()

    # the grid
    self.move_wall_grid = None


  def set_move_wall_grid(self, grid):
    """
    set grid
    """
    self.move_wall_grid = grid


  def is_moveable(self):
    """
    moveable flag
    """
    return True


  def speech_command(self, command):
    """
    speech command
    """

    # direction
    if command == 'left': self.direction_change([-1, 0])
    elif command == 'right': self.direction_change([1, 0])
    elif command == 'up': self.direction_change([0, -1])
    elif command == 'down': self.direction_change([0, 1])


  def direction_change(self, direction):
    """
    move character to position
    """

    # update move direction in Movable class
    self.update_move_direction(direction)


  def reset(self):
    """
    reset move wall
    """

    # set active
    self.is_active = True

    # grid pos
    self.grid_pos[0] = self.init_grid_pos[0]
    self.grid_pos[1] = self.init_grid_pos[1]

    # reset rect
    self.rect.x = self.grid_pos[0] * self.size[0]
    self.rect.y = self.grid_pos[1] * self.size[1]


  def grid_update(self):
    """
    update grid data
    """

    # TODO: update grid upon rect
    # handle move wall grids
    # try:
    #   self.move_wall_grid[old_pos[0], old_pos[1]] = 0
    #   self.move_wall_grid[self.grid_pos[0], self.grid_pos[1]] = 1
    # except:
    #   print("no grid stuff implemented")
    pass


  def update(self):
    """
    update movable wall moves
    """

    # not active
    if not self.is_active: return

    # move update
    self.move_update()



if __name__ == '__main__':
  """
  wall
  """

  import yaml
  from game_logic import GameLogic

  # append paths
  import sys
  sys.path.append("../")

  from classifier import Classifier
  from mic import Mic

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))

  # background color
  background_color = 255, 255, 255


  # create classifier
  classifier = Classifier(cfg_classifier=cfg['classifier'], root_path='../')

  # create mic instance
  mic = Mic(classifier=classifier, mic_params=cfg['mic_params'], is_audio_record=False)


  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(cfg['game']['screen_size'])

  # sprite groups
  all_sprites = pygame.sprite.Group()
  wall_sprites = pygame.sprite.Group()

  # create normal wall
  wall = Wall(position=(cfg['game']['screen_size'][0]//2, cfg['game']['screen_size'][1]//4))

  # create movable walls
  move_wall = MovableWall(grid_pos=[10, 10], color=(10, 100, 100), grid_move=True)
  move_wall_mic = MovableWall(grid_pos=[12, 12], color=(10, 100, 100), grid_move=True)
  move_wall_mic2 = MovableWall(grid_pos=[15, 15], color=(10, 100, 50), grid_move=True)

  # add to sprite groups
  all_sprites.add(wall, move_wall, move_wall_mic, move_wall_mic2)
  wall_sprites.add(wall)

  # obstacles
  move_wall.obstacle_sprites.add(wall_sprites, move_wall_mic, move_wall_mic2)
  move_wall_mic.obstacle_sprites.add(wall_sprites, move_wall)

  # game logic
  game_logic = GameLogic()

  # key handler
  input_handler_key = InputKeyHandler(objs=[game_logic, move_wall]) 
  input_handler_mic = InputMicHandler(objs=[move_wall_mic, move_wall_mic2], mic=mic)

  # add clock
  clock = pygame.time.Clock()

  # init stream
  mic.init_stream()

  # stream and update
  with mic.stream:

    # game loop
    while game_logic.run_loop:
      for event in pygame.event.get():

        # input handling
        game_logic.event_update(event)
        input_handler_key.event_update(event)
      
      # mic update
      input_handler_mic.update()

      # update
      all_sprites.update()
      game_logic.update()

      # fill screen
      screen.fill(background_color)

      # draw sprites
      all_sprites.draw(screen)

      # update display
      pygame.display.flip()

      # reduce framerate
      clock.tick(cfg['game']['fps'])


  # end pygame
  pygame.quit()