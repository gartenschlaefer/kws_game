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

    # init sprite dict
    self.sprite_dict = {}

    # create sprites
    self.sprite_dict.update({name: [self.spritesheet.subsurface(r) for r in rects] for name, rects in self.define_sprite_cuts().items()})

    # scale sprites
    self.sprite_dict = {name: [pygame.transform.scale(s, (s.get_width() * self.scale[0], s.get_height() * self.scale[1])) for s in surfs] for name, surfs in self.sprite_dict.items()}


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

    # jim sprite cuts
    cut_dict.update({'front': [(i*16, 0, 16, 16) for i in range(1, 2)]})
    cut_dict.update({'side-l': [(i*16, 1*16, 16, 16) for i in range(4)]})
    cut_dict.update({'side-r': [(i*16, 2*16, 16, 16) for i in range(4)]})

    return cut_dict



class SpritesheetBubbles(Spritesheet):
  """
  spritesheet class of Jim
  """

  def __init__(self, scale=(2, 2)):
    
    # Parent init
    super().__init__(file=str(pathlib.Path(__file__).parent.absolute()) + '/art/bubbles/bubbles_sprite.png', scale=scale)


  def define_sprite_cuts(self):
    """
    define individual cuts of sprites
    """

    # init cut dict
    cut_dict = {}

    # jim sprite cuts
    cut_dict.update({'question': [(i*16, 0, 16, 16) for i in range(0, 1)]})
    cut_dict.update({'rubbish': [(i*16, 0, 16, 16) for i in range(1, 2)]})

    return cut_dict



class SpritesheetSpaceshipThing(Spritesheet):
  """
  spritesheet class of Jim
  """

  def __init__(self, scale=(2, 2)):
    
    # Parent init
    super().__init__(file=str(pathlib.Path(__file__).parent.absolute()) + '/art/spaceship/spaceship_parts_spritesheet.png', scale=scale)


  def define_sprite_cuts(self):
    """
    define individual cuts of sprites
    """

    # init cut dict
    cut_dict = {}

    # jim sprite cuts
    cut_dict.update({'engine': [(i*16, 0, 16, 16) for i in range(4)]})
    cut_dict.update({'stir': [(i*16, 16, 16, 16) for i in range(4)]})

    return cut_dict



class SpritesheetIceMonster(Spritesheet):
  """
  spritesheet class of Jim
  """

  def __init__(self, scale=(2, 2)):
    
    # Parent init
    super().__init__(file=str(pathlib.Path(__file__).parent.absolute()) + '/art/ice_monster/ice_monster_spritesheet.png', scale=scale)


  def define_sprite_cuts(self):
    """
    define individual cuts of sprites
    """

    # init cut dict
    cut_dict = {}

    # jim sprite cuts
    cut_dict.update({'front': [(i*16, 0, 16, 16) for i in range(4)]})

    return cut_dict



class SpritesheetRenderer():
  """
  spritesheet renderer class
  """

  def __init__(self, anim_frame_update):

    # arguments
    self.anim_frame_update = anim_frame_update

    # frame
    self.anim_frame = 0

    # sprite index
    self.sprite_index = 0

    # define sprite dictionary
    self.sprite_dict = self.define_sprite_dictionary()

    # actual view
    self.view = list(self.sprite_dict.keys())[0]

    # subset of sprites
    self.view_sprites = self.sprite_dict[self.view]


  def get_actual_sprite(self):
    """
    get actual sprite
    """
    return self.view_sprites[self.sprite_index]


  def define_sprite_dictionary(self):
    """
    sprite sheet to dictionary (overwrite this)
    """

    # sprite dictionary
    sprite_dict = {'not_implemented': None}

    return sprite_dict


  def change_view_sprites(self, view):
    """
    view must be in the set of the sprite dict
    """

    # safety check
    if not view in self.sprite_dict.keys():
      print("view of sprite is not in list: {}".format(view))
      return

    # view update
    self.view = view

    # view sprites update
    self.view_sprites = self.sprite_dict[self.view]


  def update_spritesheet_renderer(self):
    """
    update spritesheet rendering
    """

    # frame counts
    self.anim_frame += 1

    if self.anim_frame > self.anim_frame_update:

      # update sprite index, reset anim frame
      self.sprite_index += 1
      self.anim_frame = 0

    # loop animation
    if self.sprite_index >= len(self.view_sprites): self.sprite_index = 0



if __name__ == '__main__':
  """
  test character
  """

  import yaml

  from game_logic import GameLogic
  from input_handler import InputKeyHandler

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))

  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(cfg['game']['screen_size'])

  # game logic
  game_logic = GameLogic()

  # key handler
  input_handler_key = InputKeyHandler(objs=[game_logic]) 

  # add clock
  clock = pygame.time.Clock()

  # spritesheets
  spritesheet = SpritesheetJim()
  spritesheet_bubbles = SpritesheetBubbles()
  
  screen.blit(spritesheet.sprite_dict['side-l'][0], (0, 0))
  screen.blit(spritesheet.sprite_dict['side-r'][0], (50, 50))

  screen.blit(spritesheet_bubbles.sprite_dict['question'][0], (100, 50))
  screen.blit(spritesheet_bubbles.sprite_dict['rubbish'][0], (150, 50))

  # game loop
  while game_logic.run_loop:
    for event in pygame.event.get():
      if event.type == pygame.QUIT: 
        run_loop = False

      # input handling
      game_logic.event_update(event)
      input_handler_key.event_update(event)

    # frame update
    game_logic.update()

    # update display
    pygame.display.flip()


    # reduce frame rate
    clock.tick(cfg['game']['fps'])

  # end pygame
  pygame.quit()