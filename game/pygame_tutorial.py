"""
some pygame tutorials
"""

import sys, pygame

# size of display
size = width, height = 640, 480

speed = [2, 2]
black = 0, 0, 0

# init display
screen = pygame.display.set_mode(size)

# ball init
ball = pygame.image.load("./ignore/intro_ball.gif")
ballrect = ball.get_rect()

# game loop
while 1:
	
	for event in pygame.event.get():

		# end game
		if event.type == pygame.QUIT: 

			print("quit")
			sys.exit()

		print("event: ", event)

	# movement
	ballrect = ballrect.move(speed)

	if ballrect.left < 0 or ballrect.right > width:
		speed[0] = -speed[0]

	if ballrect.top < 0 or ballrect.bottom > height:
		speed[1] = -speed[1]

	screen.fill(black)
	screen.blit(ball, ballrect)

	pygame.display.flip()  