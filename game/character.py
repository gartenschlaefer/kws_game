"""
character class
"""

import pygame

from input_handler import InputKeyHandler
from interactable import Interactable


class Character(pygame.sprite.Sprite, Interactable):
	"""
	character class
	"""

	def __init__(self, position, scale=(3, 3)):

		# MRO check
		super().__init__()

		# load image and create rect
		self.image = pygame.image.load("./art/henry_front.png").convert_alpha()
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
		self.move_speed = 2
		self.move_dir = [0, 0]

		# input handler
		self.input_handler = InputKeyHandler(self)

		# interactions
		self.obstacle_sprites = pygame.sprite.Group()
		self.thing_sprites = pygame.sprite.Group()
		self.things_collected = 0
		self.is_active = True



	def direction_change(self, direction):
		"""
		move character to position
		"""

		# apply direction change
		self.move_dir[0] += direction[0]
		self.move_dir[1] += direction[1]


	def reset(self):
		"""
		reset stuff
		"""

		self.is_active = True
		self.things = None
		self.things_collected = 0

		self.rect.x = self.init_pos[0]
		self.rect.y = self.init_pos[1]


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

		# x movement
		self.rect.x += self.move_dir[0] * self.move_speed

		# colide issue
		for wall in pygame.sprite.spritecollide(self, self.obstacle_sprites, False):

			# stand at wall
			if self.move_dir[0] > 0:
				self.rect.right = wall.rect.left

			else:
				self.rect.left = wall.rect.right


		# y movement
		self.rect.y += self.move_dir[1] * self.move_speed

		# colide issue
		for wall in pygame.sprite.spritecollide(self, self.obstacle_sprites, False):

			# stand at wall
			if self.move_dir[1] > 0:
				self.rect.bottom = wall.rect.top

			else:
				self.rect.top = wall.rect.bottom

		# interaction with things
		for thing in pygame.sprite.spritecollide(self, self.thing_sprites, True):
			self.things_collected += 1



if __name__ == '__main__':
	"""
	test character
	"""

	from color_bag import ColorBag
	from levels import LevelCharacter
	from game_logic import GameLogic

	# size of display
	screen_size = width, height = 640, 480

	# collection of game colors
	color_bag = ColorBag()

	# init pygame
	pygame.init()

	# init display
	screen = pygame.display.set_mode(screen_size)

	# level creation
	level = LevelCharacter(screen, screen_size, color_bag)

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

		# reduce framerate
		clock.tick(60)

	# end pygame
	pygame.quit()




