"""
character class
"""

import pygame
import pathlib

from input_handler import InputKeyHandler
from interactable import Interactable
from moveable import Moveable
from spritesheet import SpritesheetJim, SpritesheetBubbles


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

    # moveable init
    Moveable.__init__(self, move_sprite=self.character_sprite, move_rect=self.character_sprite.rect, move_speed=[3, 3], has_gravity=self.has_gravity, grid_move=self.grid_move)

    # save init pos
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
    self.character_sprite.rect.x = self.position[0]
    self.character_sprite.rect.y = self.position[1]


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



class CharacterSprite(pygame.sprite.Sprite):
  """
  character sprite class
  """

  def __init__(self, position, scale, anim_frame_update=3):

    # Parent init
    super().__init__()

    # arguments
    self.position = position
    self.scale = scale
    self.anim_frame_update = anim_frame_update

    # active
    self.active = True

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

    # image refs
    self.image = self.view_sprites[self.sprite_index]
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


  def change_view_sprites(self, view):
    """
    view must be in the set {"front", "side-l" "side-r"}
    """

    # safety check
    if not view in self.sprite_dict.keys():
      print("view of sprite is not in list: {}".format(view))
      return

    # view update
    self.view = view

    # view sprites update
    self.view_sprites = self.sprite_dict[self.view]


  def update(self):
    """
    update of sprite
    """

    # frame counts
    self.anim_frame += 1

    if self.anim_frame > self.anim_frame_update:

      # update sprite index, reset anim frame
      self.sprite_index += 1
      self.anim_frame = 0

    # loop animation
    if self.sprite_index >= len(self.view_sprites): self.sprite_index = 0

    # update image
    self.image = self.view_sprites[self.sprite_index]


class Henry(Character):
  """
  henry character
  """

  def define_character_sprite(self):
    """
    use jim sprite
    """
    return HenrySprite(self.position, self.scale)



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


  def actions(self, action):
    """
    other actions
    """


  def speech_command(self, command):
    """
    speech command
    """
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

