"""
building a simple grid world
"""

import pygame
import numpy as np

from wall import Wall, MovableWall
from color_bag import ColorBag


class GridWorld():
	"""
	grid world class
	"""

	def __init__(self, screen_size, color_bag, pixel_size=(20, 20)):
		"""
		create the grid world
		"""

		# variables
		self.screen_size = np.array(screen_size)
		self.pixel_size = np.array(pixel_size)
		self.color_bag = color_bag
		
		# pixel spacing
		self.grid_size = self.screen_size // self.pixel_size

		# create empty wall grid
		self.wall_grid = np.zeros(self.grid_size)
		self.wall_sprites = pygame.sprite.Group()

		# create empty movable wall grid
		self.move_wall_grid = np.zeros(self.grid_size)
		self.move_walls = []
		self.move_wall_sprites = pygame.sprite.Group()

		# active move wall
		self.act_wall = 0

		# some prints
		print("grid size: ", self.grid_size)


	def create_walls(self):
		"""
		create walls
		"""

		# TODO: destroy all walls

		# normal walls
		for i, wall_row in enumerate(self.wall_grid):
			for j, wall in enumerate(wall_row):

				# wall found
				if wall:

					#print("create wall at {}, {}".format(i, j))

					# create wall element at pixel position
					wall = Wall(position=np.array([i, j])*self.pixel_size, color=self.color_bag.wall, size=self.pixel_size)

					# add to sprite groups
					self.wall_sprites.add(wall)

		# movable walls
		for i, move_wall_row in enumerate(self.move_wall_grid):
			for j, move_wall in enumerate(move_wall_row):

				# movable wall found
				if move_wall:

					#print("create move wall at {}, {}".format(i, j))

					# create wall element at pixel position
					move_wall = MovableWall(grid_pos=[i, j], color=self.color_bag.default_move_wall, size=self.pixel_size, grid_move=True)

					# set grid
					move_wall.set_move_wall_grid(self.move_wall_grid)

					# deactivate move wall
					move_wall.is_active = False

					# wall container
					self.move_walls.append(move_wall)

					# add to sprite groups
					self.move_wall_sprites.add(move_wall)

		# collision grouping for movable walls
		for i, mw in enumerate(self.move_walls):

			# move able wall sees wall
			mw.obstacle_sprites.add(self.wall_sprites)

			# sees also moving walls
			sp = self.move_walls.copy()
			sp.pop(i)
			mw.obstacle_sprites.add(sp)

		# set one move_wall active
		try:
			self.move_walls[self.act_wall].is_active = True
			self.move_walls[self.act_wall].set_color(self.color_bag.active_move_wall)
		except:
			print("no moving walls")


	def event_move_walls(self, event):
		"""
		event handling for move walls
		"""

		for move_wall in self.move_walls:

			# handle only active wall
			if move_wall.is_active:

				# handle event
				move_wall.input_handler.handle(event)

				# event disabled wall
				if not move_wall.is_active:


					# increase index
					self.act_wall += 1

					# check if last wall
					if self.act_wall >= len(self.move_walls):
						self.act_wall = 0

					# set new active wall
					self.move_walls[self.act_wall].is_active = True

					# set colors
					self.move_walls[self.act_wall].set_color(self.color_bag.active_move_wall)
					move_wall.set_color(default_wall_color)
					break


	def event_update(self, event, run_loop):
		"""
		event update of grid world
		"""

		if event.type == pygame.QUIT: 
			run_loop = False

		# in direction
		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_ESCAPE:
				run_loop = False

		# events of move walls
		self.event_move_walls(event)

		return run_loop



def setup_level(grid_world):
	"""
	setup level
	"""

	# set walls
	grid_world.wall_grid[:, 0] = 1
	grid_world.wall_grid[5, 5] = 1

	# move walls
	grid_world.move_wall_grid[8, 8] = 1
	grid_world.move_wall_grid[10, 15] = 1
	grid_world.move_wall_grid[12, 20] = 1

	# create walls
	grid_world.create_walls()


if __name__ == '__main__':
	"""
	Main Gridworld
	"""

	# size of display
	screen_size = width, height = 640, 480

	# some vars
	run_loop = True

	# collection of game colors
	color_bag = ColorBag()
	default_wall_color = (10, 100, 100)

	# init pygame
	pygame.init()

	# init display
	screen = pygame.display.set_mode(screen_size)

	# sprite groups
	all_sprites = pygame.sprite.Group()

	# create gridworld
	grid_world = GridWorld(screen_size, color_bag)
	setup_level(grid_world)

	# add sprites
	all_sprites.add(grid_world.wall_sprites, grid_world.move_wall_sprites)

	# add clock
	clock = pygame.time.Clock()

	# game loop
	while run_loop:
		for event in pygame.event.get():

			# input handling in grid world
			run_loop = grid_world.event_update(event, run_loop)

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
