"""
moveable class
"""

import pygame
import numpy as np


class Moveable():
  """
  movement class used in character
  """

  def __init__(self, move_sprite, move_rect, move_speed=[3, 3], has_gravity=False, grid_move=False):

    # arguments
    self.move_sprite = move_sprite
    self.move_rect = move_rect
    self.move_speed = move_speed
    self.has_gravity = has_gravity
    self.grid_move = grid_move

    # move direction e.g. [1, 0]
    self.move_dir = [0, 0]
    self.move_size = [self.move_rect.width, self.move_rect.height]

    # gravity stuff
    self.is_grounded = False
    self.gravity_change = 0.3
    self.init_fall_speed = 3
    self.max_fall_speed = 6
    self.jump_force = 6

    # safety for gravity
    if grid_move: self.has_gravity = False


  def update_move_direction(self, direction):
    """
    update direction
    """

    # single movement
    if self.grid_move:
      self.move_dir[0] = direction[0]
      self.move_dir[1] = direction[1]

    # constant movement
    else:

      # apply x direction
      self.move_dir[0] += direction[0]
      self.move_dir[1] += direction[1] if not self.has_gravity else 0


  def calc_gravity(self):
    """
    gravity
    """

    # grounded condition
    if self.is_grounded:
      self.move_speed[1] = self.init_fall_speed

    # change speed according to gravity
    if self.move_speed[1] < self.max_fall_speed:
      self.move_speed[1] += 0.3

    # fix direction of gravity
    self.move_dir[1] = 1


  def jump(self):
    """
    character jump
    """

    # only if grounded
    if self.is_grounded and self.has_gravity and not self.grid_move:

      # change vertical speed
      self.move_speed[1] = -self.jump_force

      # not grounded anymore
      self.is_grounded = False


  def move_update(self):
    """
    update movable wall moves
    """

    # perform a grid move or const move
    self.move_grid() if self.grid_move else self.move_const()


  def move_const(self):
    """
    movement
    """

    # change of x
    move_change_x = self.move_dir[0] * self.move_speed[0]

    # x movement
    self.move_rect.x += move_change_x

    # collide issue
    try:
      for obst in pygame.sprite.spritecollide(self.move_sprite, self.obstacle_sprites, False):

        # stand at wall
        if move_change_x > 0: self.move_rect.right = obst.rect.left
        else: self.move_rect.left = obst.rect.right

      # y gravity
      if self.has_gravity: self.calc_gravity()

    except:
      print("no collisions implemented")

    # change of y
    move_change_y = self.move_dir[1] * self.move_speed[1]

    # y movement
    self.move_rect.y += move_change_y

    # grounded false
    self.is_grounded = False

    # collide issue
    try:
      for obst in pygame.sprite.spritecollide(self.move_sprite, self.obstacle_sprites, False):
        
        # stand at wall
        if move_change_y > 0:

          # stop atop
          self.move_rect.bottom = obst.rect.top

          # grounded condition
          self.is_grounded = True

        else:

          # stop with head hit
          self.move_rect.top = obst.rect.bottom

          # no upward movement anymore (jump)
          if self.has_gravity:
            self.move_speed[1] = 0

    except:
      print("no collisions implemented")


  def move_grid(self):
    """
    move in grid to position
    """

    # update position if changed
    if np.any(self.move_dir):

      # update actual pos
      self.move_rect.x += self.move_dir[0] * self.move_size[0]
      self.move_rect.y += self.move_dir[1] * self.move_size[1]

      try:
        #collide issue
        for obst in pygame.sprite.spritecollide(self.move_sprite, self.obstacle_sprites, False):

          # hit an obstacle
          if obst:

            # old position again
            self.move_rect.x -= self.move_dir[0] * self.move_size[0]
            self.move_rect.y -= self.move_dir[1] * self.move_size[1]

      except:
        print("no collisions implemented")

      # reset move direction
      self.move_dir = [0, 0]