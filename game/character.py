"""
character class
"""

import pygame
import pathlib

from input_handler import InputKeyHandler
from interactable import Interactable


class Character(pygame.sprite.Sprite, Interactable):
	"""
	character class
	"""

	def __init__(self, position, scale=(3, 3), is_gravity=False):

		# MRO check
		super().__init__()

		self.position = position
		self.scale = scale
		self.is_gravity = is_gravity

		# load image and create rect
		self.image = pygame.image.load(str(pathlib.Path(__file__).parent.absolute()) + "/art/henry_front.png").convert_alpha()
		self.rect = self.image.get_rect()

		# proper scaling
		self.image = pygame.transform.scale(self.image, (self.rect.width * scale[0], self.rect.height * scale[1]))
		self.rect = self.image.get_rect()

		# save init pos
		self.init_pos = position

		# set rect position
		self.rect.x = position[0]
		self.rect.y = position[1]

		# speed and move dir
		self.move_speed = [3, 3]
		self.move_dir = [0, 0]

		# gravity stuff
		self.gravity_change = 0
		self.is_grounded = False
		self.max_fall_speed = 6
		self.init_fall_speed = 3
		self.jump_force = 6

		# input handler
		self.input_handler = InputKeyHandler(self)

		# interactions
		self.obstacle_sprites = pygame.sprite.Group()
		self.thing_sprites = pygame.sprite.Group()
		self.things_collected = 0
		self.is_active = True


	def set_position(self, position, is_init_pos=False):
		"""
		set position absolute
		"""

		# set internal pos
		self.position = position

		# also set initial position
		if is_init_pos:
			self.init_pos = position

		# set rect
		self.rect.x = self.position[0]
		self.rect.y = self.position[1]


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

		# determine direction determined by move speed
		self.move_dir[1] = 1


	def jump(self):
		"""
		character jump
		"""

		# only if grounded
		if self.is_grounded:

			# change vertical speed
			self.move_speed[1] = -self.jump_force

			# not grounded anymore
			self.is_grounded = False


	def direction_change(self, direction):
		"""
		move character to position
		"""

		# apply x direction
		self.move_dir[0] += direction[0]

		if self.is_gravity:
			return

		# apply y direction
		self.move_dir[1] += direction[1]


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

		self.is_active = True
		self.is_grounded = False
		self.things = None
		self.things_collected = 0

		# set init position
		self.set_position(self.init_pos)


	def event_update(self, event):
		"""
		event update for character
		"""

		# event handling
		self.input_handler.handle(event)



	def update(self):
		"""
		update character
		"""

		# not active
		if not self.is_active:
			return

		# change of x
		move_change_x = self.move_dir[0] * self.move_speed[0]

		# x movement
		self.rect.x += move_change_x

		# collide issue
		for obst in pygame.sprite.spritecollide(self, self.obstacle_sprites, False):

			# stand at wall
			if move_change_x > 0:
				self.rect.right = obst.rect.left

			else:
				self.rect.left = obst.rect.right


		# y gravity
		if self.is_gravity:

			# calculate gravity
			self.calc_gravity()


		# change of y
		move_change_y = self.move_dir[1] * self.move_speed[1]

		# y movement
		self.rect.y += move_change_y

		# grounded false
		self.is_grounded = False

		# collide issue
		for obst in pygame.sprite.spritecollide(self, self.obstacle_sprites, False):
			
			# stand at wall
			if move_change_y > 0:

				# stop atop
				self.rect.bottom = obst.rect.top

				# grounded condition
				self.is_grounded = True

			else:

				# stop with head hit
				self.rect.top = obst.rect.bottom

				# no upward movement anymore
				self.move_speed[1] = 0

		# interaction with things
		for thing in pygame.sprite.spritecollide(self, self.thing_sprites, True):
			self.things_collected += 1



if __name__ == '__main__':
	"""
	test character
	"""

	import yaml

	from color_bag import ColorBag
	from levels import LevelCharacter
	from game_logic import GameLogic

	# yaml config file
	cfg = yaml.safe_load(open("../config.yaml"))

	# collection of game colors
	color_bag = ColorBag()

	# init pygame
	pygame.init()

	# init display
	screen = pygame.display.set_mode(cfg['game']['screen_size'])

	# level creation
	level = LevelCharacter(screen, cfg['game']['screen_size'], color_bag)

	# game logic
	game_logic = GameLogic()

	# add clock
	clock = pygame.time.Clock()

	# game loop
	while game_logic.run_loop:
		for event in pygame.event.get():
			if event.type == pygame.QUIT: 
				run_loop = False

			# input handling
			game_logic.event_update(event)
			level.event_update(event)

		# frame update
		game_logic.update()
		level.update()

		# update display
		pygame.display.flip()

		# reduce frame rate
		clock.tick(cfg['game']['fps'])

	# end pygame
	pygame.quit()




