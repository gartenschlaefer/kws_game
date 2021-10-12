"""
character class
"""

import pygame
import pathlib

from interactable import Interactable
from moveable import Moveable
from spritesheet import SpritesheetIceMonster, SpritesheetRenderer


class Enemy(Interactable, Moveable):
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
    self.enemy_sprite = EnemySprite(self.position, self.scale)

    # moveable
    Moveable.__init__(self, move_sprite=self.enemy_sprite, move_rect=self.enemy_sprite.rect, move_speed=[3, 3], has_gravity=self.has_gravity, grid_move=self.grid_move)

    # save initial position
    self.init_pos = position

    # interactions
    self.obstacle_sprites = pygame.sprite.Group()
    self.hit_sprites = pygame.sprite.Group()

    # hit
    self.hit_enemy = False

    # active
    self.is_active = True

    # sprites
    self.sprites = pygame.sprite.Group()

    # add character sprite
    self.sprites.add(self.enemy_sprite)

    # move to direction
    self.update_move_direction((1, 0))


  def obstacle_action(self, direction):
    """
    obstacle action
    """
    if direction[0] > 0: self.update_move_direction((-2, 0))
    else: self.update_move_direction((2, 0))


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

    # set rect
    self.enemy_sprite.rect.x = self.position[0]
    self.enemy_sprite.rect.y = self.position[1]


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
    # ToDo implement...
    # # update sprite view
    # if self.move_dir[0] < 0: self.enemy_sprite.change_view_sprites("side-l")
    # elif self.move_dir[0] > 0: self.enemy_sprite.change_view_sprites("side-r")
    # else: self.enemy_sprite.change_view_sprites("front")
    pass


  def reset(self):
    """
    reset stuff
    """

    # reset
    self.is_active = True
    self.is_grounded = False
    self.hit_enemy = False

    # set initial position
    self.set_position(self.init_pos)


  def hit_collide(self, hit_sprite):
    """
    hit the enemy
    """
    self.hit_enemy = True


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
    [self.hit_collide(hit) for hit in pygame.sprite.spritecollide(self.enemy_sprite, self.hit_sprites, False)]

    # update sprites
    self.sprites.update()


  def draw(self):
    """
    draw all sprites of the character
    """
    self.sprites.draw(self.surf)



class EnemySprite(pygame.sprite.Sprite, SpritesheetRenderer):
  """
  enemy sprite class
  """

  def __init__(self, position, scale, anim_frame_update=6):

    # parent init
    super().__init__()

    # arguments
    self.position = position
    self.scale = scale
    self.anim_frame_update = anim_frame_update

    # spritesheet renderer init
    SpritesheetRenderer.__init__(self, anim_frame_update=anim_frame_update)

    # image refs
    self.image = self.get_actual_sprite()
    self.rect = self.image.get_rect()

    # set rect position
    self.rect.x, self.rect.y = self.position[0], self.position[1]


  def define_sprite_dictionary(self):
    """
    sprite sheet to dictionary
    """

    # init sprite sheet
    self.spritesheet = SpritesheetIceMonster(scale=self.scale)

    return self.spritesheet.sprite_dict


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
  test enemy
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

  # enemy
  enemy = Enemy(surf=screen, position=(40, 40), scale=(2, 2), has_gravity=True, grid_move=False)
  enemy.obstacle_sprites = level.interactable_dict['character'].obstacle_sprites
  enemy.hit_sprites.add(level.interactable_dict['character'].character_sprite)

  # add enemy to level
  level.interactable_dict.update({'enemy': enemy})

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

