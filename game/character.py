"""
character class
"""

import pygame
import numpy as np

from wall import Wall
from input_handler import InputKeyHandler


class Character(pygame.sprite.Sprite):
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

		# set rect position
		self.rect.x = position[0]
		self.rect.y = position[1]

		# speed and move dir
		self.move_speed = 2
		self.move_dir = [0, 0]

		# interactions
		self.walls = None
		self.is_active = True

		# input handler
		self.input_handler = InputKeyHandler(self)


	def direction_change(self, direction):
		"""
		move character to position
		"""

		# apply direction change
		self.move_dir[0] += direction[0]
		self.move_dir[1] += direction[1]


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
		for wall in pygame.sprite.spritecollide(self, self.walls, False):

			# stand at wall
			if self.move_dir[0] > 0:
				self.rect.right = wall.rect.left

			else:
				self.rect.left = wall.rect.right


		# y movement
		self.rect.y += self.move_dir[1] * self.move_speed

		# colide issue
		for wall in pygame.sprite.spritecollide(self, self.walls, False):

			# stand at wall
			if self.move_dir[1] > 0:
				self.rect.bottom = wall.rect.top

			else:
				self.rect.top = wall.rect.bottom


if __name__ == '__main__':
	"""
	test character
	"""

	# size of display
	size = width, height = 640, 480

	# some vars
	run_loop = True
	background_color = 255, 255, 255

	# init pygame
	pygame.init()

	# init display
	screen = pygame.display.set_mode(size)

	# sprite groups
	all_sprites = pygame.sprite.Group()
	wall_sprites = pygame.sprite.Group()

	# create the character
	henry = Character(position=(width//2, height//2))

	wall = Wall(position=(width//2, height//4))

	# add to sprite groups
	all_sprites.add(henry, wall)
	wall_sprites.add(wall)

	# henry sees walls
	henry.walls = wall_sprites

	# add clock
	clock = pygame.time.Clock()

	# game loop
	while run_loop:
		for event in pygame.event.get():
			if event.type == pygame.QUIT: 
				run_loop = False

			# input handling of henry
			henry.input_handler.handle(event)

		# update sprites
		all_sprites.update()

		# fill screen
		screen.fill(background_color)

		# draw sprites
		all_sprites.draw(screen)

		# update display
		pygame.display.flip()

		# reduce framerate
		clock.tick(60)


	# end pygame
	pygame.quit()




