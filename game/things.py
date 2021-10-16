"""
things class
"""

import pygame
import pathlib

from spritesheet import SpritesheetRenderer, SpritesheetSpaceshipThing, SpritesheetSpaceship
from interactable import Interactable


class ThingOrigin(pygame.sprite.Sprite):
  """
  thing originally
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


  def set_position(self, position):
    """
    set position absolute
    """

    # set internal pos
    self.position = position

    # set rect
    self.rect.x = self.position[0]
    self.rect.y = self.position[1]



class Thing(Interactable):
  """
  thing base class
  """

  def __init__(self, surf, position, scale=(2, 2)):

    # arguments
    self.surf = surf
    self.position = position
    self.scale = scale

    # save initial position
    self.init_pos = position

    # interactions
    self.is_active = True

    # sprites
    self.sprites = pygame.sprite.Group()

    # add character sprite
    self.sprites.add(self.define_sprites())


  def change_view(self, view):
    """
    change view
    """
    [s.change_view_sprites(view=view) for s in self.sprites]


  def define_sprites(self):
    """
    define sprites (overwrite this)
    """

    from character import CharacterSprite

    return [CharacterSprite(position=self.position, scale=self.scale, anim_frame_update=5)]


  def set_active(self, active):
    """
    set active
    """
    self.is_active = active


  def set_position(self, position, is_init_pos=False):
    """
    set position absolute
    """

    # set internal pos
    self.position = position

    # also set initial position
    if is_init_pos: self.init_pos = position

    # set sprite position
    [s.set_position(self.position) for s in self.sprites]


  def reset(self):
    """
    reset stuff
    """

    # reset
    self.is_active = True

    # set init position
    self.set_position(self.init_pos)


  def update(self):
    """
    update character
    """

    # not active
    if not self.is_active: return

    # update sprites
    self.sprites.update()


  def draw(self):
    """
    draw all sprites of the character
    """
    self.sprites.draw(self.surf)



class SpaceshipThing(Thing):
  """
  spaceship thing class
  """

  def __init__(self, surf, position, scale=(2, 2), thing_type='engine'):

    # arguments
    self.thing_type = thing_type

    # parent init
    super().__init__(surf=surf, position=position, scale=scale)


  def define_sprites(self):
    """
    define sprites (overwrite this)
    """
    return [SpaceshipThingSprite(self.position, self.scale, thing_type=self.thing_type, anim_frame_update=20)]



class Spaceship(Thing):
  """
  spaceship thing class
  """

  def __init__(self, surf, position, scale=(2, 2), thing_type='whole'):

    # arguments
    self.thing_type = thing_type

    # parent init
    super().__init__(surf=surf, position=position, scale=scale)


  def define_sprites(self):
    """
    define sprites (overwrite this)
    """
    return [SpaceshipSprite(self.position, self.scale, thing_type=self.thing_type, anim_frame_update=20)]



class SpaceshipThingSprite(pygame.sprite.Sprite, SpritesheetRenderer):
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

    # change sprite thing
    self.change_view_sprites(view=self.thing_type)

    # image refs
    self.image = self.get_actual_sprite()
    self.rect = self.image.get_rect()

    # set rect position
    self.rect.x, self.rect.y = self.position[0], self.position[1]


  def set_position(self, position):
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

    return self.spritesheet.sprite_dict


  def update(self):
    """
    update of sprite
    """

    # update sprite sheet renderer
    self.update_spritesheet_renderer()

    # update
    self.image = self.get_actual_sprite()



class SpaceshipSprite(SpaceshipThingSprite):
  """
  spaceship
  """

  def define_sprite_dictionary(self):
    """
    sprite sheet to dictionary
    """

    # init sprite sheet
    self.spritesheet = SpritesheetSpaceship(scale=self.scale)

    return self.spritesheet.sprite_dict



if __name__ == '__main__':
  """
  things
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

  # add a spaceship to the level
  #levels[0].interactable_dict.update({'new_thing': Spaceship(surf=screen, position=(100, 100), scale=(2, 2), thing_type='empty')})

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