"""
character class
"""

import pygame
import numpy as np

from input_handler import InputHandler


class Wall(pygame.sprite.Sprite):
	"""
	wall class
	"""

	def __init__(self, position, color=(10, 200, 200), size=(20, 20)):

		# MRO check
		super().__init__()

		# ball init
		self.image = pygame.surface.Surface(size)
		self.rect = self.image.get_rect()

		# set rect position
		self.rect.x = position[0]
		self.rect.y = position[1]

		self.image.fill(color)


class MovableWall(Wall):
	"""
	a movable wall
	"""

	def __init__(self, grid_pos, color=(10, 200, 200), size=(20, 20), grid_move=False):

		# vars
		self.grid_pos = grid_pos
		self.size = size
		self.grid_move = grid_move

		# MRO check
		super().__init__(np.array(grid_pos)*size, color, size)

		# input handler
		self.input_handler = InputHandler(self, handler_type='key_stroke_dir', grid_move=grid_move)

		# speed and move dir
		self.move_speed = 2
		self.move_dir = [0, 0]

		# interactions
		self.walls = None
		self.is_active = True
		self.position = [self.rect.x, self.rect.y]

		# the grid
		self.move_wall_grid = None


	def set_move_wall_grid(self, grid):
		"""
		let the moveable grid
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


	def move_const(self):
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


	def move_grid(self):
		"""
		move in grid to position
		"""

		# not active
		if not self.is_active:
			return

		# update position if changed
		if np.any(self.move_dir):

			# handle move wall grids
			try:
				self.move_wall_grid[self.grid_pos[0], self.grid_pos[1]] = 0
				self.move_wall_grid[self.grid_pos[0]+self.move_dir[0], self.grid_pos[1]+self.move_dir[1]] = 1
			except:
				print("no grid stuff")

			# new pos
			self.grid_pos[0] += self.move_dir[0]  
			self.grid_pos[1] += self.move_dir[1]

			# reset move dir
			self.move_dir = [0, 0]

			# update actual pos
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
	test character
	"""

	# size of display
	size = width, height = 640, 480

	# some vars
	run_loop = True
	background_color = 255, 255, 255

	# grid move
	grid_move = True

	# init pygame
	pygame.init()

	# init display
	screen = pygame.display.set_mode(size)

	# sprite groups
	all_sprites = pygame.sprite.Group()
	wall_sprites = pygame.sprite.Group()

	# create the character
	move_wall = MovableWall(grid_pos=[2, 2], color=(10, 100, 100), grid_move=grid_move)

	wall = Wall(position=(width//2, height//4))

	# add to sprite groups
	all_sprites.add(move_wall, wall)
	wall_sprites.add(wall)

	# henry sees walls
	move_wall.walls = wall_sprites

	# add clock
	clock = pygame.time.Clock()

	# game loop
	while run_loop:
		for event in pygame.event.get():
			if event.type == pygame.QUIT: 
				run_loop = False

			# input handling of movable wall
			move_wall.input_handler.handle(event)

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