"""
building a simple grid world
"""

import pygame
import numpy as np

from wall import Wall


class GridWorld():
	"""
	grid world class
	"""

	def __init__(self, screen_size, pixel_size=(20, 20)):
		"""
		create the grid world
		"""

		# variables
		self.screen_size = np.array(screen_size)
		self.pixel_size = np.array(pixel_size)
		
		# pixel spacing
		self.grid_size = self.screen_size // self.pixel_size

		# create empty wall grid
		self.wall_grid = np.zeros(self.grid_size)
		self.wall_sprites = pygame.sprite.Group()

		# some prints
		print("grid size: ", self.grid_size)


	def create_walls(self):
		"""
		create walls
		"""

		# TODO: destroy all walls

		for i, wall_row in enumerate(self.wall_grid):

			for j, wall in enumerate(wall_row):

				# wall found
				if wall:

					print("create wall at {}, {}".format(i, j))
					# create wall element at pixel position
					wall = Wall(position=np.array([i, j])*self.pixel_size, size=self.pixel_size)

					# add to sprite groups
					self.wall_sprites.add(wall)



if __name__ == '__main__':
	"""
	Main Gridworld
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

	# create gridworld
	grid_world = GridWorld(size)

	# set walls
	grid_world.wall_grid[:, 0] = 1
	grid_world.create_walls()
	all_sprites.add(grid_world.wall_sprites)

	# add clock
	clock = pygame.time.Clock()

	# game loop
	while run_loop:
		for event in pygame.event.get():
			if event.type == pygame.QUIT: 
				run_loop = False


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
