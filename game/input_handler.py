"""
input handler class
"""

import pygame


class InputHandler():
	"""
	Input Handler class
	"""

	def __init__(self, obj, grid_move=False):

		# object to be handled
		self.obj = obj

		# grid handling
		self.grid_move = grid_move


	def handle(self, event):
		"""
		handle object
		"""
		pass



class InputKeyHandler(InputHandler):
	"""
	Keyboard handler
	"""

	def __init__(self, obj, grid_move=False):

		# init of father class
		super().__init__(obj, grid_move)


	def handle(self, event):
		"""
		handle keyboard inputs
		"""

		# key down
		if event.type == pygame.KEYDOWN:

			if event.key == pygame.K_LEFT:
				self.obj.direction_change([-1, 0])
			elif event.key == pygame.K_RIGHT:
				self.obj.direction_change([1, 0])
			if event.key == pygame.K_UP:
				self.obj.direction_change([0, -1])
			elif event.key == pygame.K_DOWN:
				self.obj.direction_change([0, 1])

			# jump button
			if event.key == pygame.K_SPACE:
				self.obj.action_key()

			if event.key == pygame.K_RETURN:
				self.obj.enter_key()

			if event.key == pygame.K_ESCAPE:
				self.obj.esc_key()

		# key up
		elif event.type == pygame.KEYUP and not self.grid_move:

			if event.key == pygame.K_LEFT:
				self.obj.direction_change([1, 0])
			elif event.key == pygame.K_RIGHT:
				self.obj.direction_change([-1, 0])
			if event.key == pygame.K_UP:
				self.obj.direction_change([0, 1])
			elif event.key == pygame.K_DOWN:
				self.obj.direction_change([0, -1])



class InputMicHandler(InputHandler):
	"""
	Microphone handler
	"""

	def __init__(self, obj, mic, grid_move=False):

		# init of father class
		super().__init__(obj, grid_move)

		self.mic = mic


	def handle(self, event):
		"""
		handle keyboard inputs
		"""

		# get command
		command = self.mic.update_read_command()

		# interpret command
		if command is not None:

			print("yey command: ", command)

			# direction
			if command == 2:
				self.obj.direction_change([-1, 0])
			elif command == 3:
				self.obj.direction_change([1, 0])
			elif command == 4:
				self.obj.direction_change([0, -1])
			elif command == 0:
				self.obj.direction_change([0, 1])

			# deactivate
			elif command == 1:

				# action
				self.obj.action_key()
