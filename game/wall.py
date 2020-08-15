"""
character class
"""

import pygame
import numpy as np


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