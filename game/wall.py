"""
character class
"""

import pygame
import numpy as np

from input_handler import InputKeyHandler, InputMicHandler
from interactable import Interactable


class Wall(pygame.sprite.Sprite):
	"""
	wall class
	"""

	def __init__(self, position, color=(10, 200, 200), size=(20, 20)):

		# MRO check
		super().__init__()

		# vars
		self.position = position
		self.color = color
		self.size = size

		# wall init
		self.image = pygame.surface.Surface(self.size)
		self.rect = self.image.get_rect()

		# set rect position
		self.rect.x = self.position[0]
		self.rect.y = self.position[1]

		self.image.fill(color)


	def set_color(self, color):
		"""
		set the color
		"""

		self.image.fill(color)



class MovableWall(Wall, Interactable):
	"""
	a movable wall
	"""

	def __init__(self, grid_pos, color=(10, 200, 200), size=(20, 20), grid_move=False, mic_control=False, mic=None):

		# vars
		self.grid_pos = grid_pos
		self.grid_move = grid_move

		# MRO check
		super().__init__(np.array(grid_pos)*size, color, size)

		# input handler
		if mic_control:
			self.input_handler = InputMicHandler(self, mic=mic, grid_move=grid_move)

		else:
			self.input_handler = InputKeyHandler(self, grid_move=grid_move)

		# speed and move dir
		self.move_speed = 2
		self.move_dir = [0, 0]

		# interactions
		self.obstacle_sprites = pygame.sprite.Group()
		self.is_active = True
		
		# save init pos
		self.init_pos = self.position
		self.init_grid_pos = self.grid_pos.copy()

		# the grid
		self.move_wall_grid = None


	def set_move_wall_grid(self, grid):
		"""
		set grid
		"""

		self.move_wall_grid = grid


	def direction_change(self, direction):
		"""
		move character to position
		"""

		# single movement
		if self.grid_move:
			self.move_dir[0] = direction[0]
			self.move_dir[1] = direction[1]

		# constant movement change
		else:
			self.move_dir[0] += direction[0]
			self.move_dir[1] += direction[1]


	def action_key(self):
		"""
		if action key is pressed
		"""

		self.is_active = not self.is_active


	def move_const(self):
		"""
		update character
		"""

		# not active
		if not self.is_active:
			return

		# x movement
		self.rect.x += self.move_dir[0] * self.move_speed

		# collide issue
		for obst in pygame.sprite.spritecollide(self, self.obstacle_sprites, False):

			# stand at wall
			if self.move_dir[0] > 0:
				self.rect.right = obst.rect.left

			else:
				self.rect.left = obst.rect.right

		# y movement
		self.rect.y += self.move_dir[1] * self.move_speed

		# collide issue
		for obst in pygame.sprite.spritecollide(self, self.obstacle_sprites, False):

			# stand at wall
			if self.move_dir[1] > 0:
				self.rect.bottom = obst.rect.top

			else:
				self.rect.top = obst.rect.bottom


	def move_grid(self):
		"""
		move in grid to position
		"""

		# not active
		if not self.is_active:
			return

		# update position if changed
		if np.any(self.move_dir):

			# save old pos
			old_pos = self.grid_pos.copy()

			# new pos
			self.grid_pos[0] += self.move_dir[0]  
			self.grid_pos[1] += self.move_dir[1]

			# update actual pos
			self.rect.x = self.grid_pos[0] * self.size[0]
			self.rect.y = self.grid_pos[1] * self.size[1]

			try:
				#collide issue
				for obst in pygame.sprite.spritecollide(self, self.obstacle_sprites, False):

					# hit an obstacle
					if obst:

						# old position again
						self.grid_pos = old_pos
						self.rect.x = self.grid_pos[0] * self.size[0]
						self.rect.y = self.grid_pos[1] * self.size[1]
			except:
				print("no collisions implemented")

			# handle move wall grids
			try:
				self.move_wall_grid[old_pos[0], old_pos[1]] = 0
				self.move_wall_grid[self.grid_pos[0], self.grid_pos[1]] = 1
			except:
				print("no grid stuff implemented")

			# reset move direction
			self.move_dir = [0, 0]


	def reset(self):
		"""
		reset move wall
		"""

		# set active
		self.is_active = True

		# grid pos
		self.grid_pos[0] = self.init_grid_pos[0]
		self.grid_pos[1] = self.init_grid_pos[1]

		# reset rect
		self.rect.x = self.grid_pos[0] * self.size[0]
		self.rect.y = self.grid_pos[1] * self.size[1]


	def update(self):
		"""
		update movable wall moves
		"""

		if self.grid_move:

			# perform a grid move
			self.move_grid()

		else:
			
			# move constantly
			self.move_const()


if __name__ == '__main__':
	"""
	wall
	"""

	import yaml

	# append paths
	import sys
	sys.path.append("../")

	from classifier import Classifier
	from mic import Mic
	from game_logic import GameLogic
	from path_collector import PathCollector

	# yaml config file
	cfg = yaml.safe_load(open("../config.yaml"))

	# init path collector
	path_coll = PathCollector(cfg, root_path='.')

	# some vars
	run_loop = True
	background_color = 255, 255, 255

	# grid move
	grid_move = True

	# init pygame
	pygame.init()

	# init display
	screen = pygame.display.set_mode(cfg['game']['screen_size'])

	# sprite groups
	all_sprites = pygame.sprite.Group()
	wall_sprites = pygame.sprite.Group()

	# params
	fs = 16000

	# window and hop size
	N, hop = int(cfg['feature_params']['N_s'] * cfg['feature_params']['fs']), int(cfg['feature_params']['hop_s'] * cfg['feature_params']['fs'])

	# create classifier
	classifier = Classifier(path_coll=path_coll, verbose=True)

	# create mic instance
	mic = Mic(classifier=classifier, feature_params=cfg['feature_params'], mic_params=cfg['mic_params'], is_audio_record=False)

	# create normal wall
	wall = Wall(position=(cfg['game']['screen_size'][0]//2, cfg['game']['screen_size'][1]//4))

	# create movable walls
	move_wall = MovableWall(grid_pos=[10, 10], color=(10, 100, 100), grid_move=grid_move, mic_control=False)
	move_wall_mic = MovableWall(grid_pos=[12, 12], color=(10, 100, 100), grid_move=grid_move, mic_control=True, mic=mic)

	# add to sprite groups
	all_sprites.add(wall, move_wall, move_wall_mic)
	wall_sprites.add(wall)

	# henry sees walls
	move_wall.obstacle_sprites.add(wall_sprites, move_wall_mic)
	move_wall_mic.obstacle_sprites.add(wall_sprites, move_wall)

	# game logic
	game_logic = GameLogic()

	# add clock
	clock = pygame.time.Clock()

	# stream and update
	with mic.stream:

		# game loop
		while game_logic.run_loop:
			for event in pygame.event.get():

				# input handling
				game_logic.event_update(event)
				move_wall.input_handler.handle(event)
			
			# mic update
			move_wall_mic.input_handler.handle(None)

			# update
			all_sprites.update()
			game_logic.update()

			# fill screen
			screen.fill(background_color)

			# draw sprites
			all_sprites.draw(screen)

			# update display
			pygame.display.flip()

			# reduce framerate
			clock.tick(cfg['game']['fps'])


	# end pygame
	pygame.quit()