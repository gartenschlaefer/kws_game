"""
building a simple grid world
"""

import pygame
import numpy as np

from wall import Wall, MovableWall


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

		# create empty movable wall grid
		self.move_wall_grid = np.zeros(self.grid_size)
		self.move_walls = []
		self.move_wall_sprites = pygame.sprite.Group()

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

		# movable walls
		for i, move_wall_row in enumerate(self.move_wall_grid):
			for j, move_wall in enumerate(move_wall_row):

				# movable wall found
				if move_wall:

					print("create move wall at {}, {}".format(i, j))

					# create wall element at pixel position
					move_wall = MovableWall(grid_pos=[i, j], color=(10, 100, 100), size=self.pixel_size, grid_move=True)

					# set grid
					move_wall.set_move_wall_grid(self.move_wall_grid)

					self.move_walls.append(move_wall)

					# move able wall sees wall
					move_wall.walls = self.wall_sprites

					# add to sprite groups
					self.move_wall_sprites.add(move_wall)


	def event_move_walls(self, event):
		"""
		event handling for move walls
		"""

		for move_wall in self.move_walls:

			# TODO: Handle only one wall
			move_wall.input_handler.handle(event)


	def event_update(self, event):
		"""
		event update of grid world
		"""

		# events of move walls
		self.event_move_walls(event)


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
	grid_world.wall_grid[5, 5] = 1
	grid_world.move_wall_grid[8, 8] = 1
	grid_world.move_wall_grid[10, 15] = 1
	grid_world.move_wall_grid[12, 20] = 1
	grid_world.create_walls()

	# set only one moveable_wall active
	for w in grid_world.move_walls:
		w.is_active = False

	# set one active
	grid_world.move_walls[0].is_active = True

	# add sprites
	all_sprites.add(grid_world.wall_sprites, grid_world.move_wall_sprites)

	# add clock
	clock = pygame.time.Clock()

	# game loop
	while run_loop:
		for event in pygame.event.get():
			if event.type == pygame.QUIT: 
				run_loop = False

			# input handling of henry
			grid_world.event_update(event)

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
