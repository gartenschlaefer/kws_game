"""
building a simple grid world
"""

import pygame
import numpy as np

from wall import Wall, MovableWall
from color_bag import ColorBag
from interactable import Interactable

class GridWorld(Interactable):
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

		# mic control
		if self.mic is None:
			self.mic_control = False
		else:
			self.mic_control = True
		
		# pixel spacing
		self.grid_size = self.screen_size // self.pixel_size

		# create empty wall grid
		self.wall_grid = np.zeros(self.grid_size)
		self.wall_sprites = pygame.sprite.Group()

		# create empty movable wall grid
		self.move_wall_grid = np.zeros(self.grid_size)
		self.move_wall_sprites = pygame.sprite.Group()

		# move wall container
		self.move_walls = []

		# active move wall
		self.act_wall = 0


	def grid_to_pos(self, grid_pos):
		"""
		transform grid to position
		"""

		return (grid_pos[0] * self.pixel_size[0], grid_pos[1] * self.pixel_size[1])


	def create_walls(self):
		"""
		create walls
		"""

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
					move_wall = MovableWall(grid_pos=[i, j], color=self.color_bag.default_move_wall, size=self.pixel_size, grid_move=True, mic_control=self.mic_control, mic=self.mic)

					# set grid
					move_wall.set_move_wall_grid(self.move_wall_grid)

					# wall container
					self.move_walls.append(move_wall)

					# add to sprite groups
					self.move_wall_sprites.add(move_wall)

		# init move walls
		self.move_walls_init()


	def move_walls_init(self):
		"""
		init move walls
		"""

		# collision grouping for movable walls
		for i, move_wall in enumerate(self.move_walls):

			# reset moving walls
			move_wall.reset()

			# deactivate move wall
			move_wall.is_active = False
			move_wall.set_color(self.color_bag.default_move_wall)

			# move able wall sees wall
			move_wall.obstacle_sprites.add(self.wall_sprites)

			# sees also moving walls
			sp = self.move_walls.copy()
			sp.pop(i)
			move_wall.obstacle_sprites.add(sp)

		# set one move wall active
		if self.move_walls:

			# set active
			self.move_walls[self.act_wall].is_active = True
			self.move_walls[self.act_wall].set_color(self.color_bag.active_move_wall)


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


	def reset(self):
		"""
		reset grid world
		"""

		# active move wall
		self.act_wall = 0

		# init move walls again
		self.move_walls_init()


	def event_update(self, event):
		"""
		event update of grid world
		"""

		# events of move walls
		if self.mic is None:
			self.move_walls_update(event)


	def update(self):
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
	from levels import LevelMoveWalls
	from game_logic import GameLogic


	# size of display
	screen_size = width, height = 640, 480

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
	classifier = Classifier(file='../models/fstride_c-5.npz', verbose=False) 

	# create mic instance
	mic = Mic(fs=fs, N=N, hop=hop, classifier=classifier)



	# level setup
	level = LevelMoveWalls(screen, screen_size, color_bag, mic)

	# game logic
	game_logic = GameLogic()

	# add clock
	clock = pygame.time.Clock()

	# mic stream and update
	with mic.stream:

		# game loop
		while game_logic.run_loop:
			for event in pygame.event.get():

				# event handling
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
