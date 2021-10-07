"""
things class
"""

import pygame
import pathlib

from spritesheet import SpritesheetRenderer, SpritesheetSpaceshipThing


class Thing(pygame.sprite.Sprite):
  """
  character class
  """

  def __init__(self, position, scale=(2, 2)):

    # parent init
    super().__init__()

    # vars
    self.position = position
    self.scale = scale

    # load image and create rect
    self.image = pygame.image.load(str(pathlib.Path(__file__).parent.absolute()) + "/art/thing/thing.png").convert_alpha()
    self.rect = self.image.get_rect()

    # proper scaling
    self.image = pygame.transform.scale(self.image, (self.rect.width * scale[0], self.rect.height * scale[1]))
    self.rect = self.image.get_rect()

    # set rect position
    self.rect.x = position[0]
    self.rect.y = position[1]

    # interactions
    self.is_active = True


  def set_position(self, position, is_init_pos=False):
    """
    set position absolute
    """

    # set internal pos
    self.position = position

    # set rect
    self.rect.x = self.position[0]
    self.rect.y = self.position[1]



class SpaceshipThing(pygame.sprite.Sprite, SpritesheetRenderer):
  """
  spaceship thing
  """

  def __init__(self, position, scale, thing_type='engine', anim_frame_update=20):

    # parent init
    super().__init__()

    # arguments
    self.position = position
    self.scale = scale
    self.thing_type = thing_type
    self.anim_frame_update = anim_frame_update

    # spritesheet renderer init
    SpritesheetRenderer.__init__(self, anim_frame_update=anim_frame_update)

    # active
    self.active = True

    # image refs
    self.image = self.get_actual_sprite()
    self.rect = self.image.get_rect()

    # set rect position
    self.rect.x, self.rect.y = self.position[0], self.position[1]

    # change sprite thing
    self.change_view_sprites(view=self.thing_type)


  def set_position(self, position, is_init_pos=False):
    """
    set position absolute
    """

    # set internal pos
    self.position = position

    # set rect
    self.rect.x = self.position[0]
    self.rect.y = self.position[1]


  def define_sprite_dictionary(self):
    """
    sprite sheet to dictionary
    """

    # init sprite sheet
    self.spritesheet = SpritesheetSpaceshipThing(scale=self.scale)
      
    # sprite dict
    sprite_dict = self.spritesheet.sprite_dict

    return sprite_dict


  def update(self):
    """
    update of sprite
    """

    # update sprite sheet renderer
    self.update_spritesheet_renderer()

    # update
    self.image = self.get_actual_sprite()


if __name__ == '__main__':
  """
  test character
  """
  
  import yaml

  from levels import LevelThings, Level_01, LevelHandler

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))

  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(cfg['game']['screen_size'])

  # level creation
  levels = [LevelThings(screen, cfg['game']['screen_size']), Level_01(screen, cfg['game']['screen_size'])]

  # level handler
  level_handler = LevelHandler(levels=levels)

  # add clock
  clock = pygame.time.Clock()

  # game loop
  while level_handler.runs():
    for event in pygame.event.get():

      # input handling
      level_handler.event_update(event)

    # frame update
    level_handler.update()

    # update display
    pygame.display.flip()

    # reduce framerate
    clock.tick(cfg['game']['fps'])

  # end pygame
  pygame.quit()