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
		self.walls = None
		self.is_active = True
		self.things = None
		self.things_collected = 0


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


		# interaction with things
		if self.things is not None:
			for thing in pygame.sprite.spritecollide(self, self.things, True):
				self.things_collected += 1



if __name__ == '__main__':
	"""
	test character
	"""

	from wall import Wall
	from grid_world import GridWorld
	from color_bag import ColorBag
	from levels import setup_level_square


	# size of display
	screen_size = width, height = 640, 480

	# some vars
	run_loop = True

	# collection of game colors
	color_bag = ColorBag()

	# init pygame
	pygame.init()

	# init display
	screen = pygame.display.set_mode(screen_size)

	# sprite groups
	all_sprites = pygame.sprite.Group()
	wall_sprites = pygame.sprite.Group()

	# create the character
	henry = Character(position=(width//2, height//2), scale=(2, 2))

	# create gridworld
	grid_world = GridWorld(screen_size, color_bag)
	setup_level_square(grid_world)

	# add to sprite groups
	all_sprites.add(henry, grid_world.wall_sprites)

	# henry sees walls
	henry.walls = grid_world.wall_sprites

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
		screen.fill(color_bag.background)

		# draw sprites
		all_sprites.draw(screen)

		# update display
		pygame.display.flip()

		# reduce framerate
		clock.tick(60)


	# end pygame
	pygame.quit()




