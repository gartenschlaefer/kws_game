"""
character class
"""

import pygame
import pathlib

from input_handler import InputKeyHandler
from interactable import Interactable
from moveable import Moveable
from spritesheet import SpritesheetJim, SpritesheetBubbles, SpritesheetRenderer


class Character(Interactable, Moveable):
  """
  character class
  """

  def __init__(self, surf, position, scale=(3, 3), has_gravity=True, grid_move=False):

    # arguments
    self.surf = surf
    self.position = position
    self.scale = scale
    self.has_gravity = has_gravity
    self.grid_move = grid_move

    # character sprite
    self.character_sprite = self.define_character_sprite()

    # moveable
    Moveable.__init__(self, move_sprite=self.character_sprite, move_rect=self.character_sprite.rect, move_speed=[3, 3], has_gravity=self.has_gravity, grid_move=self.grid_move)

    # save initial position
    self.init_pos = position

    # interactions
    self.obstacle_sprites = pygame.sprite.Group()
    self.thing_sprites = pygame.sprite.Group()
    self.things_collected = 0
    self.is_active = True

    # sprites
    self.sprites = pygame.sprite.Group()

    # add character sprite
    self.sprites.add(self.character_sprite)


  def set_active(self, active):
    """
    set active
    """
    self.is_active = active


  def define_character_sprite(self):
    """
    define the character for child classes intended
    """
    return CharacterSprite(self.position, self.scale)


  def set_position(self, position, is_init_pos=False):
    """
    set position absolute
    """

    # set internal pos
    self.position = position

    # also set initial position
    if is_init_pos: self.init_pos = position

    # set rect
    self.character_sprite.set_position(self.position)


  def is_moveable(self):
    """
    moveable flag
    """
    return True


  def direction_change(self, direction):
    """
    move character to position
    """

    # update move direction in Movable class
    self.update_move_direction(direction)


  def view_update(self):
    """
    update view upon direction
    """

    # update sprite view
    if self.move_dir[0] < 0: self.character_sprite.change_view_sprites("side-l")
    elif self.move_dir[0] > 0: self.character_sprite.change_view_sprites("side-r")
    else: self.character_sprite.change_view_sprites("front")


  def action_key(self):
    """
    if action key is pressed
    """

    # do a jump
    self.jump()


  def reset(self):
    """
    reset stuff
    """

    # reset
    self.is_active = True
    self.is_grounded = False
    self.things = None
    self.things_collected = 0

    # set init position
    self.set_position(self.init_pos)


  def update(self):
    """
    update character
    """

    # not active
    if not self.is_active: return

    # move player with movement class
    self.move_update()

    # update of character view
    self.view_update()

    # interaction with things
    for thing in pygame.sprite.spritecollide(self.character_sprite, self.thing_sprites, True): self.things_collected += 1

    # update sprites
    self.sprites.update()


  def draw(self):
    """
    draw all sprites of the character
    """
    self.sprites.draw(self.surf)



class Henry(Character):
  """
  henry character
  """

  def define_character_sprite(self):
    """
    use jim sprite
    """
    return HenrySprite(self.position, self.scale)



class Jim(Character):
  """
  Jim the shovelnaut
  """

  def __init__(self, surf, position, scale=(3, 3), has_gravity=True, grid_move=False):

    # Parent init
    super().__init__(surf, position, scale, has_gravity, grid_move)

    # bubble sprite
    self.bubble_sprite = BubbleSprite(position, scale)

    # deactivate bubble sprite
    self.bubble_sprite.active = False

    # active frames
    self.bubble_active_frames = 45
    self.bubble_frame = self.bubble_active_frames

    # activate bubble
    self.activate_bubble_sprite(view='question', activate=False)


  def activate_bubble_sprite(self, view='question', activate=True):
    """
    activate bubble
    """

    # change view
    self.bubble_sprite.change_view_sprites(view)

    # add or remove sprite
    self.sprites.add(self.bubble_sprite) if activate else self.sprites.remove(self.bubble_sprite)
    self.bubble_sprite.active = activate
    self.bubble_frame = self.bubble_active_frames


  def define_character_sprite(self):
    """
    use jim sprite
    """
    return JimSprite(self.position, self.scale)


  def speech_command(self, command):
    """
    speech command
    """

    # not active
    if not self.is_active: return

    # bubbles
    if command == '_noise': self.activate_bubble_sprite(view='rubbish', activate=True)
    elif command == '_mixed': self.activate_bubble_sprite(view='question', activate=True)


  def update(self):
    """
    update character
    """

    # not active
    if not self.is_active: return

    # reduce active count
    if self.bubble_sprite.active: 
      self.bubble_frame -= 1
      if not self.bubble_frame: self.activate_bubble_sprite(activate=False)

    # move player with movement class
    self.move_update()

    # bubble possition
    self.bubble_sprite.rect.x = self.character_sprite.rect.x - 26
    self.bubble_sprite.rect.y = self.character_sprite.rect.y - 30

    # update of character view
    self.view_update()

    # interaction with things
    for thing in pygame.sprite.spritecollide(self.character_sprite, self.thing_sprites, True): self.things_collected += 1

    # update character sprite
    self.sprites.update()



class CharacterSprite(pygame.sprite.Sprite, SpritesheetRenderer):
  """
  character sprite class
  """

  def __init__(self, position, scale, view_type='front', anim_frame_update=5):

    # parent init
    super().__init__()

    # arguments
    self.position = position
    self.scale = scale
    self.view_type = view_type
    self.anim_frame_update = anim_frame_update

    # spritesheet renderer init
    SpritesheetRenderer.__init__(self, anim_frame_update=anim_frame_update)

    # change sprite thing
    self.change_view_sprites(view=self.view_type)

    # image refs
    self.image = self.get_actual_sprite()
    self.rect = self.image.get_rect()

    # set rect position
    self.rect.x, self.rect.y = self.position[0], self.position[1]


  def define_sprite_dictionary(self):
    """
    sprite sheet to dictionary
    """

    # sprite dictionary
    sprite_dict = {'front': [pygame.Surface(size=(16, 16))], 'side-r': [pygame.Surface(size=(16, 16))], 'side-l': [pygame.Surface(size=(16, 16))]}

    # fill
    sprite_dict['front'][0].fill((100, 100, 0))
    sprite_dict['side-r'][0].fill((50, 100, 50))
    sprite_dict['side-l'][0].fill((50, 50, 100))

    return sprite_dict


  def set_position(self, position, is_init_pos=False):
    """
    set position absolute
    """

    # set internal pos
    self.position = position

    # set rect
    self.rect.x = self.position[0]
    self.rect.y = self.position[1]


  def update(self):
    """
    update of sprite
    """

    # update sprite sheet renderer
    self.update_spritesheet_renderer()

    # update
    self.image = self.get_actual_sprite()



class HenrySprite(CharacterSprite):
  """
  henry sprite
  """

  def define_sprite_dictionary(self):
    """
    sprite sheet to dictionary
    """

    # root for sprites
    sprite_root_path = str(pathlib.Path(__file__).parent.absolute()) + '/art/henry/'

    # image file names
    image_file_names = ['henry_front.png', 'henry_side-1.png', 'henry_side-2.png', 'henry_side-3.png']

    # index for sprites infos
    view_index = {'front':(0, 1), 'side-r':(1, 4), 'side-l':(4, 7)}

    # actual sprites as image arrays
    sprites = [pygame.image.load(sprite_root_path + s).convert_alpha() for s in image_file_names]
    
    # scale sprites
    sprites = [pygame.transform.scale(s, (s.get_width() * self.scale[0], s.get_height() * self.scale[1])) for s in sprites]
    
    # extend right sprites
    sprites.extend([pygame.transform.flip(s, True, False) for s in sprites[view_index['side-r'][0]:view_index['side-r'][1]]])
    
    # sprite dict
    sprite_dict = {k: sprites[v[0]:v[1]] for k, v in view_index.items()}

    return sprite_dict



class JimSprite(CharacterSprite):
  """
  character sprite class
  """

  def define_sprite_dictionary(self):
    """
    sprite sheet to dictionary
    """

    # init sprite sheet
    self.spritesheet = SpritesheetJim(scale=self.scale)
      
    # sprite dict
    sprite_dict = self.spritesheet.sprite_dict

    return sprite_dict



class BubbleSprite(CharacterSprite):
  """
  character sprite class
  """

  def __init__(self, position, scale, view_type='question', anim_frame_update=5):

    # parent init
    super().__init__(position, scale, view_type=view_type, anim_frame_update=anim_frame_update)


  def define_sprite_dictionary(self):
    """
    sprite sheet to dictionary
    """

    # init sprite sheet
    self.spritesheet = SpritesheetBubbles(scale=self.scale)
      
    # sprite dict
    sprite_dict = self.spritesheet.sprite_dict

    return sprite_dict



if __name__ == '__main__':
  """
  test character
  """

  import yaml
  from levels import LevelCharacter

  # append paths
  import sys
  sys.path.append("../")

  from classifier import Classifier
  from mic import Mic

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))

  # create classifier
  classifier = Classifier(cfg_classifier=cfg['classifier'], root_path='../')

  # create mic instance
  mic = Mic(classifier=classifier, mic_params=cfg['mic_params'], is_audio_record=False)

  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(cfg['game']['screen_size'])

  # level creation
  level = LevelCharacter(screen, cfg['game']['screen_size'], mic=mic)

  # add clock
  clock = pygame.time.Clock()

  # init stream
  mic.init_stream()

  # stream and update
  with mic.stream:

    # game loop
    while level.runs():
      for event in pygame.event.get():
        if event.type == pygame.QUIT: 
          run_loop = False

        # level event update
        level.event_update(event)

      # frame update
      level.update()

      # update display
      pygame.display.flip()

      # reduce frame rate
      clock.tick(cfg['game']['fps'])

  # end pygame
  pygame.quit()

