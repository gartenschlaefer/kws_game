"""
spritesheets for ghosts
"""

import pygame
import numpy as np
import pathlib


class Spritesheet():
  """
  spritesheet class
  """

  def __init__(self, file, scale=(2, 2)):
    
    # arguments
    self.file = file
    self.scale = scale

    # load spritesheet
    self.spritesheet = pygame.image.load(self.file).convert_alpha()

    # get rect
    self.rect = self.spritesheet.get_rect()

    # scale sprites
    #self.spritesheet = pygame.transform.scale(self.spritesheet, (self.rect.width * self.scale[0], self.rect.height * self.scale[1]))

    # init sprite dict
    self.sprite_dict = {}

    # create sprites
    self.create_sprites()

    # scale sprites
    self.sprite_dict = {name: [pygame.transform.scale(s, (s.get_width() * self.scale[0], s.get_height() * self.scale[1])) for s in surfs] for name, surfs in self.sprite_dict.items()}


  def create_sprites(self):
    """
    create sprites
    """
    self.sprite_dict.update({name: [self.spritesheet.subsurface(r) for r in rects] for name, rects in self.define_sprite_cuts().items()})


  def define_sprite_cuts(self):
    """
    define individual cuts of sprites
    """
    return {}



class SpritesheetJim(Spritesheet):
  """
  spritesheet class of Jim
  """

  def __init__(self, scale=(2, 2)):
    
    # Parent init
    super().__init__(file=str(pathlib.Path(__file__).parent.absolute()) + '/art/shovelnaut/shovelnaut_spritesheet.png', scale=scale)


  def define_sprite_cuts(self):
    """
    define individual cuts of sprites
    """

    # init cut dict
    cut_dict = {}

    # jim
    cut_dict.update({'left': [(i*16, 0, 16, 16) for i in range(2)]})
    cut_dict.update({'right': [(i*16, 16, 16, 16) for i in range(2)]})

    return cut_dict



if __name__ == '__main__':
  """
  test character
  """

  import yaml

  from game_logic import GameLogic

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))

  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(cfg['game']['screen_size'])

  # game logic
  game_logic = GameLogic()

  # add clock
  clock = pygame.time.Clock()

  # sprite sheet
  #spritesheet = Spritesheet(file='./art/shovelnaut/shovelnaut_spritesheet.png')
  spritesheet = SpritesheetJim()
  
  screen.blit(spritesheet.sprite_dict['left'][0], (0, 0))
  screen.blit(spritesheet.sprite_dict['right'][0], (50, 50))

  # game loop
  while game_logic.run_loop:
    for event in pygame.event.get():
      if event.type == pygame.QUIT: 
        run_loop = False

      # input handling
      game_logic.event_update(event)

    # frame update
    game_logic.update()

    # update display
    pygame.display.flip()


    # reduce frame rate
    clock.tick(cfg['game']['fps'])

  # end pygame
  pygame.quit()