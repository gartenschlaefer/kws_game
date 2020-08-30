"""
levels
"""

import pygame


def setup_level_square(grid_world):
  """
  setup level
  """

  # set walls
  grid_world.wall_grid[:, 0] = 1
  grid_world.wall_grid[:, -1] = 1
  grid_world.wall_grid[0, :] = 1
  grid_world.wall_grid[-1, :] = 1

  # wall in the middle
  grid_world.wall_grid[7, 7] = 1

  # create walls
  grid_world.create_walls()


def setup_level_move_wall(grid_world):
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
  levels
  """

  from color_bag import ColorBag
  from grid_world import GridWorld

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


  # create gridworld
  grid_world = GridWorld(screen_size, color_bag)
  setup_level_square(grid_world)

  # add sprites
  all_sprites.add(grid_world.wall_sprites, grid_world.move_wall_sprites)

  # add clock
  clock = pygame.time.Clock()



  # game loop
  while run_loop:
    for event in pygame.event.get():

      # input handling in grid world
      run_loop = grid_world.event_update(event, run_loop)

    # frame update
    grid_world.update()

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




  

