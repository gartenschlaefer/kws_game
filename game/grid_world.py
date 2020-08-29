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

	def __init__(self, screen_size, color_bag, mic=None, pixel_size=(20, 20)):
		"""
		create the grid world
		"""

		# variables
		self.screen_size = np.array(screen_size)
		self.pixel_size = np.array(pixel_size)
		self.color_bag = color_bag
		self.mic = mic
		
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

				# normal wall found
				if wall:

					# create wall element at pixel position
					wall = Wall(position=np.array([i, j])*self.pixel_size, color=self.color_bag.wall, size=self.pixel_size)

					# add to sprite groups
					self.wall_sprites.add(wall)

		# movable walls
		for i, move_wall_row in enumerate(self.move_wall_grid):
			for j, move_wall in enumerate(move_wall_row):

				# movable wall found
				if move_wall:

					# create wall element at pixel position
					move_wall = MovableWall(grid_pos=[i, j], color=self.color_bag.default_move_wall, size=self.pixel_size, grid_move=True, mic_control=True, mic=self.mic)

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


	def move_walls_update(self, event=None):
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
					move_wall.set_color(self.color_bag.default_move_wall)
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
		if self.mic is None:
			self.move_walls_update(event)

		return run_loop


	def frame_update(self):
		"""
		frame update
		"""

		if self.mic is not None:
			self.move_walls_update()


if __name__ == '__main__':
	"""
	Main Gridworld
	"""

	# append paths
	import sys
	sys.path.append("../")

	from classifier import Classifier
	from mic import Mic
	from levels import setup_level_move_wall


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


	# --
	# mic

	# params
	fs = 16000

	# window and hop size
	N, hop = int(0.025 * fs), int(0.010 * fs)

	# create classifier
	classifier = Classifier(file='../ignore/models/best_models/fstride_c-5.npz', root_dir='.')  

	# create mic instance
	mic = Mic(fs=fs, N=N, hop=hop, classifier=classifier)


	# create gridworld
	grid_world = GridWorld(screen_size, color_bag, mic)
	setup_level_move_wall(grid_world)

	# add sprites
	all_sprites.add(grid_world.wall_sprites, grid_world.move_wall_sprites)

	# add clock
	clock = pygame.time.Clock()

	# mic stream and update
	with mic.stream:

		# game loop
		while run_loop:
			for event in pygame.event.get():

				# input handling in grid world
				run_loop = grid_world.event_update(event, run_loop)

			# frame update
			grid_world.frame_update()

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
