"""
character class
"""

import pygame


class Character(pygame.sprite.Sprite):
	"""
	character
	"""

	def __init__(self, position, orient):

		# MRO check
		super().__init__()

		# position
		self.position = position
		self.orient = orient

		# ball init
		self.image = pygame.image.load("./art/henry_front.png").convert_alpha()
		self.rect = self.image.get_rect()

		# set rect position
		self.rect.x = self.position[0]
		self.rect.y = self.position[1]


	def move(self, position):
		"""
		move character to position
		"""
		pass


	def change_orient(self, orient):
		"""
		change characters orientation
		"""
		pass



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

	# all sprite list
	all_sprite_list = pygame.sprite.Group()

	# create the character
	henry = Character(position=(width//2, height//2), orient=0)

	# add henry to sprites
	all_sprite_list.add(henry)

	# game loop
	while run_loop:
		for event in pygame.event.get():
			if event.type == pygame.QUIT: 
				run_loop = False

		# fill screen
		screen.fill(background_color)

		# draw sprites
		all_sprite_list.draw(screen)

		# update display
		pygame.display.flip()


	# end pygame
	pygame.quit()




