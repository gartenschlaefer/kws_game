"""
some polymorphism and inheritance examples in python
"""

class Animal():
	"""
	Animal Base class
	"""

	def __new__(cls, change_type=None):
		"""
		new instance
		"""

		print("animal new class: ", cls)

		if change_type == 'cat':

			for child_cls in cls.__subclasses__():
				print("child: ", child_cls)
				if child_cls.__name__ == 'Cat':
					return super().__new__(Cat)

		elif change_type == 'dog':

			for child_cls in cls.__subclasses__():
				print("child: ", child_cls)
				if child_cls.__name__ == 'Dog':
					return super().__new__(Cat)

		return super().__new__(cls)


	def __init__(self, change_type=None):

		self.id = 'animal'
		self.type = 'none'
		print("animal init")


	def make_sound(self):
		"""
		make sound
		"""

		print("Animal Sound: ...")



class Cat(Animal):
	"""
	Cat class
	"""

	#def __new__(cls, change_type=None)
	#print("animal new")

	def __init__(self, change_type=None):

		# parent class init
		super().__init__(change_type)

		self.type = 'cat'

		print("cat init")


	def make_sound(self):
		"""
		make sound
		"""

		print("Cat Sound: Miaowwwh")


	def growl(self):
		"""
		cat growls
		"""

		print("Cat Growl: Purrrrh")



class Dog(Animal):
	"""
	Cat class
	"""

	def __init__(self, change_type=None):

		# parent class init
		super().__init__(change_type)

		self.type = 'dog'


	def make_sound(self):
		"""
		make sound
		"""

		print("Dog Sound: Wuffff")


	def growl(self):
		"""
		cat growls
		"""

		print("Dog Growl: Grrrrr")


if __name__ == '__main__':
	"""
	main
	"""

	animal = Animal()
	cat = Cat()

	print("animal: ", animal)
	print("cat: ", cat)

	print("animal id: ", animal.id)
	print("cat id: ", cat.id)

	print("animal type: ", animal.type)
	print("cat type: ", cat.type)

	animal.make_sound()
	cat.make_sound()
	cat.growl()

	print("\nanimal cat")

	animal_cat = Animal(change_type='cat')
	animal_cat.make_sound()
	animal_cat.growl()


