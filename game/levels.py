"""
levels
"""

import pygame

from interactable import Interactable
from color_bag import ColorBag
from canvas import CanvasWin


class Level(Interactable):
  """
  level class
  """

  def __init__(self, screen, screen_size):

    # colors
    self.screen = screen
    self.screen_size = screen_size
    self.color_bag = ColorBag()

    # sprites
    self.all_sprites = pygame.sprite.Group()

    # interactables
    self.interactable_dict = {}


  def setup_level(self):
    """
    setup level
    """
    pass


  def reset(self):
    """
    reset level
    """
    for interactable in self.interactable_dict.values(): interactable.reset()


  def event_update(self, event):
    """
    event update
    """
    for interactable in self.interactable_dict.values(): interactable.event_update(event)


  def update(self):
    """
    update
    """

    # interactables
    for interactable in self.interactable_dict.values(): interactable.update()

    # update sprites
    self.all_sprites.update()

    # fill screen
    self.screen.fill(self.color_bag.background)

    # draw interactables
    for interactable in self.interactable_dict.values(): interactable.draw()

    # draw sprites
    self.all_sprites.draw(self.screen)



class LevelMic(Level):
  """
  level class
  """

  def __init__(self, screen, screen_size, mic):

    from mic_bar import MicBar

    # parent class init
    super().__init__(screen, screen_size)

    # arguments
    self.mic = mic

    # append interactable
    self.interactable_dict.update({'mic_bar': MicBar(self.screen, self.mic, position=(200, 200), bar_size=(50, 150), scale_margin=(50, 40))})



class LevelGrid(Level):
  """
  level with grid
  """

  def __init__(self, screen, screen_size, mic=None):

    from grid_world import GridWorld

    # parent class init
    super().__init__(screen, screen_size)

    # new vars
    self.mic = mic

    # create gridworld
    self.grid_world = GridWorld(self.screen_size, self.color_bag, self.mic)

    # setup
    self.setup_level()

    # append interactable
    #self.interactables.append(self.grid_world)
    self.interactable_dict.update({'grid_world': self.grid_world})

    # sprites
    self.all_sprites.add(self.grid_world.wall_sprites, self.grid_world.move_wall_sprites)


  def setup_level(self):
    """
    setup level
    """

    # set walls
    self.setup_wall_edge()

    # create walls
    self.grid_world.create_walls()


  def setup_wall_edge(self):
    """
    limit edges
    """

    # set walls
    self.grid_world.wall_grid[:, 0] = 1
    self.grid_world.wall_grid[:, -1] = 1
    self.grid_world.wall_grid[0, :] = 1
    self.grid_world.wall_grid[-1, :] = 1



class LevelSquare(LevelGrid):
  """
  level square
  """

  def __init__(self, screen, screen_size):

    # parent class init
    super().__init__(screen, screen_size)


  def setup_level(self):
    """
    setup level
    """

    # set walls
    self.setup_wall_edge()

    # wall in the middle
    self.grid_world.wall_grid[7, 7] = 1

    # create walls
    self.grid_world.create_walls()



class LevelMoveWalls(LevelGrid):
  """
  level with moving walls
  """

  def __init__(self, screen, screen_size, mic=None):

    # parent class init
    super().__init__(screen, screen_size, mic)


  def setup_level(self):
    """
    setup level
    """

    # set walls
    self.setup_wall_edge()

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

  def __init__(self, screen, screen_size, mic=None):

    # parent class init
    super().__init__(screen, screen_size, mic)

    from character import Character

    # create the character
    self.henry = Character(position=(self.screen_size[0]//2, self.screen_size[1]//2), scale=(2, 2), has_gravity=True, grid_move=False)
    self.henry.obstacle_sprites.add(self.grid_world.wall_sprites, self.grid_world.move_wall_sprites)

    # add interactable
    self.interactable_dict.update({'henry': self.henry})

    # add to sprites
    self.all_sprites.add(self.henry.character_sprite)


  def setup_level(self):
    """
    setup level
    """

    # set walls
    self.setup_wall_edge()

    # wall in the middle
    self.grid_world.wall_grid[20:25, 20:24] = 1
    self.grid_world.wall_grid[10:15, 20] = 1

    # create walls
    self.grid_world.create_walls()



class LevelThings(LevelCharacter):
  """
  level with character
  """

  def __init__(self, screen, screen_size, mic=None):

    # parent class init
    super().__init__(screen, screen_size, mic)

    from things import Thing

    # create thing
    self.thing = Thing(position=self.grid_world.grid_to_pos([22, 18]), scale=(2, 2))

    # add to sprites
    self.all_sprites.add(self.thing)
    self.henry.thing_sprites.add(self.thing)

    # determine position
    self.henry.set_position(self.grid_world.grid_to_pos([10, 10]), is_init_pos=True)

    # add interactables
    self.interactable_dict.update({'win_canvas': CanvasWin(self.screen)})


  def win(self):
    """
    win condition for level
    """

    # deactivate henry
    self.henry.is_active = False

    # activate win canvas
    self.interactable_dict['win_canvas'].enabled = True


  def reset(self):
    """
    reset level
    """

    self.grid_world.reset()
    self.henry.reset()

    # add to sprites
    self.all_sprites.add(self.thing)
    self.henry.thing_sprites.add(self.thing)

    # interactables reset
    for interactable in self.interactable_dict.values(): interactable.reset()



class Level_01(LevelThings):
  """
  first actual level
  """

  def __init__(self, screen, screen_size, mic=None):

    # parent class init
    super().__init__(screen, screen_size, mic)

    # determine start position
    self.henry.set_position(self.grid_world.grid_to_pos([5, 20]), is_init_pos=True)
    self.thing.set_position(self.grid_world.grid_to_pos([22, 18]), is_init_pos=True)


  def setup_level(self):
    """
    setup level
    """

    # set walls
    self.setup_wall_edge()

    # wall in the middle
    self.grid_world.wall_grid[20:25, 20:23] = 1
    self.grid_world.wall_grid[20:25, 16] = 1
    self.grid_world.wall_grid[25, 16:23] = 1

    self.grid_world.wall_grid[10:15, 20:23] = 1

    # move walls
    self.grid_world.move_wall_grid[20, 17] = 1
    self.grid_world.move_wall_grid[20, 18] = 1
    self.grid_world.move_wall_grid[20, 19] = 1

    # create walls
    self.grid_world.create_walls()



class Level_02(LevelThings):
  """
  first actual level
  """

  def __init__(self, screen, screen_size, mic=None):

    # parent class init
    super().__init__(screen, screen_size, mic)

    # determine start position
    self.henry.set_position(self.grid_world.grid_to_pos([22, 20]), is_init_pos=True)
    self.thing.set_position(self.grid_world.grid_to_pos([2, 5]), is_init_pos=True)


  def setup_level(self):
    """
    setup level
    """

    # set walls
    self.setup_wall_edge()

    # wall in the middle
    self.grid_world.wall_grid[27:, 19] = 1
    self.grid_world.wall_grid[19:22, 15] = 1
    self.grid_world.wall_grid[11:14, 11] = 1

    self.grid_world.wall_grid[:5, 7] = 1

    # move walls
    self.grid_world.move_wall_grid[29, 22] = 1
    self.grid_world.move_wall_grid[27, 18] = 1
    self.grid_world.move_wall_grid[19, 16] = 1
    self.grid_world.move_wall_grid[4, 8] = 1

    # create walls
    self.grid_world.create_walls()



if __name__ == '__main__':
  """
  levels
  """
  import yaml

  from game_logic import GameLogic, ThingsGameLogic
  from text import Text

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))


  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(cfg['game']['screen_size'])

  # level creation
  levels = [Level_01(screen, cfg['game']['screen_size']), Level_02(screen, cfg['game']['screen_size']), LevelSquare(screen, cfg['game']['screen_size']), LevelMoveWalls(screen, cfg['game']['screen_size'])]

  # choose level
  level = levels[0]

  # game logic with dependencies
  game_logic = ThingsGameLogic(level, levels)

  # add clock
  clock = pygame.time.Clock()


  # game loop
  while game_logic.run_loop:
    for event in pygame.event.get():

      # input handling
      game_logic.event_update(event)
      level.event_update(event)

    # frame update
    level = game_logic.update()
    level.update()

    # update display
    pygame.display.flip()

    # reduce framerate
    clock.tick(cfg['game']['fps'])

  # end pygame
  pygame.quit()




  

