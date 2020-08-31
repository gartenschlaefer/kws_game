"""
levels
"""

import pygame


class Level():
  """
  level class
  """

  def __init__(self, screen, screen_size, color_bag):

    # colors
    self.screen = screen
    self.screen_size = screen_size
    self.color_bag = color_bag

    # sprites
    self.all_sprites = pygame.sprite.Group()


  def setup_level(self):
    """
    setup level
    """
    pass


  def event_update(self):
    """
    event update
    """
    pass


  def update(self):
    """
    update
    """

    # update sprites
    self.all_sprites.update()

    # fill screen
    self.screen.fill(self.color_bag.background)

    # draw sprites
    self.all_sprites.draw(screen)



class LevelGrid(Level):
  """
  level with grid
  """

  def __init__(self, screen, screen_size, color_bag, mic=None):

    from grid_world import GridWorld

    # parent class init
    super().__init__(screen, screen_size, color_bag)

    # new vars
    self.mic = mic

    # create gridworld
    self.grid_world = GridWorld(self.screen_size, self.color_bag, self.mic)

    # setup
    self.setup_level(self.grid_world)

    # sprites
    self.all_sprites.add(self.grid_world.wall_sprites, self.grid_world.move_wall_sprites)


  def setup_level(self, grid_world):
    """
    setup level
    """

    # set walls
    self.grid_world.wall_grid[:, 0] = 1
    self.grid_world.wall_grid[:, -1] = 1
    self.grid_world.wall_grid[0, :] = 1
    self.grid_world.wall_grid[-1, :] = 1

    # create walls
    self.grid_world.create_walls()


  def event_update(self, event):
    """
    event update
    """
    
    # grid world update
    self.grid_world.event_update(event)


  def update(self):
    """
    update
    """

    # grid world update
    self.grid_world.update()

    # update sprites
    self.all_sprites.update()

    # fill screen
    self.screen.fill(self.color_bag.background)

    # draw sprites
    self.all_sprites.draw(self.screen)




class LevelSquare(LevelGrid):
  """
  level square
  """

  def __init__(self, screen, screen_size, color_bag):

    # parent class init
    super().__init__(screen, screen_size, color_bag)


  def setup_level(self, grid_world):
    """
    setup level
    """

    # set walls
    self.grid_world.wall_grid[:, 0] = 1
    self.grid_world.wall_grid[:, -1] = 1
    self.grid_world.wall_grid[0, :] = 1
    self.grid_world.wall_grid[-1, :] = 1

    # wall in the middle
    self.grid_world.wall_grid[7, 7] = 1

    # create walls
    self.grid_world.create_walls()



class LevelMoveWalls(LevelGrid):
  """
  level with moving walls
  """

  def __init__(self, screen, screen_size, color_bag, mic=None):

    # parent class init
    super().__init__(screen, screen_size, color_bag, mic)


  def setup_level(self, grid_world):
    """
    setup level
    """

    # set walls
    self.grid_world.wall_grid[:, 0] = 1
    self.grid_world.wall_grid[:, -1] = 1
    self.grid_world.wall_grid[0, :] = 1
    self.grid_world.wall_grid[-1, :] = 1
    self.grid_world.wall_grid[5, 5] = 1

    # move walls
    self.grid_world.move_wall_grid[8, 8] = 1
    self.grid_world.move_wall_grid[10, 15] = 1
    self.grid_world.move_wall_grid[12, 20] = 1

    # create walls
    self.grid_world.create_walls()



class LevelCharacter(LevelGrid):
  """
  level with character
  """

  def __init__(self, screen, screen_size, color_bag, mic=None):

    # parent class init
    super().__init__(screen, screen_size, color_bag, mic)

    from character import Character

    # create the character
    self.henry = Character(position=(self.screen_size[0]//2, self.screen_size[1]//2), scale=(2, 2))
    self.henry.obstacle_sprites.add(self.grid_world.wall_sprites, self.grid_world.move_wall_sprites)

    # add to sprites
    self.all_sprites.add(self.henry)


  def event_update(self, event):
    """
    event update
    """
    
    # grid world update
    self.henry.event_update(event)
    self.grid_world.event_update(event)



class LevelThings(LevelCharacter):
  """
  level with character
  """

  def __init__(self, screen, screen_size, color_bag, mic=None):

    # parent class init
    super().__init__(screen, screen_size, color_bag, mic)

    from things import Thing

    # create thing
    self.thing = Thing(position=(100, 100), scale=(2, 2))

    # add to sprites
    self.all_sprites.add(self.thing)
    self.henry.thing_sprites.add(self.thing)



if __name__ == '__main__':
  """
  levels
  """

  from color_bag import ColorBag
  from grid_world import GridWorld
  from game_logic import GameLogic

  # size of display
  screen_size = width, height = 640, 480

  # collection of game colors
  color_bag = ColorBag()

  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(screen_size)


  # level creation
  #level = LevelSquare(screen, screen_size, color_bag)
  level = LevelMoveWalls(screen, screen_size, color_bag)

  # game logic
  game_logic = GameLogic()

  # add clock
  clock = pygame.time.Clock()


  # game loop
  while game_logic.run_loop:
    for event in pygame.event.get():

      # input handling
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




  

