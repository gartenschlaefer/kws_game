"""
levels
"""

import pygame

from interactable import Interactable
from color_bag import ColorBag
from canvas import CanvasWin
from input_handler import InputKeyHandler, InputMicHandler
from game_logic import GameLogic, ThingsGameLogic
from character import Character, Henry, Jim
from text import Text


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

    # interactable dict
    self.interactable_dict = {} 

    # game logic
    self.define_game_logic()

    # key handler
    self.interactable_dict.update({'input_key_handler': InputKeyHandler(objs=[self.interactable_dict['game_logic']])})


  def define_game_logic(self):
    """
    define interactables
    """
    self.interactable_dict.update({'game_logic': GameLogic()})


  def runs(self):
    """
    check if level runs
    """
    return self.interactable_dict['game_logic'].run_loop


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

    # draw sprites
    self.all_sprites.draw(self.screen)

    # draw interactables
    for interactable in self.interactable_dict.values(): interactable.draw()



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

    # append to key handler
    self.interactable_dict['input_key_handler'].objs.append(self.interactable_dict['mic_bar'])



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
    self.grid_world = GridWorld(self.screen_size, self.color_bag)

    # setup
    self.setup_level()

    # append interactable
    self.interactable_dict.update({'grid_world': self.grid_world})

    # sprites
    self.all_sprites.add(self.grid_world.wall_sprites, self.grid_world.move_wall_sprites)

    # append input handler
    self.interactable_dict['input_key_handler'].objs.append(self.grid_world) if mic is None else self.interactable_dict.update({'input_mic_handler': InputMicHandler(objs=[self.interactable_dict['grid_world']], mic=self.mic)})


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

    # create the character
    #self.character = Character(surf=self.screen, position=(self.screen_size[0]//2, self.screen_size[1]//2), scale=(2, 2), has_gravity=True, grid_move=False)
    #self.character = Henry(surf=self.screen, position=(self.screen_size[0]//2, self.screen_size[1]//2), scale=(2, 2), has_gravity=True, grid_move=False)
    self.character = Jim(surf=self.screen, position=(self.screen_size[0]//2, self.screen_size[1]//2), scale=(2, 2), has_gravity=True, grid_move=False)
    self.character.obstacle_sprites.add(self.grid_world.wall_sprites, self.grid_world.move_wall_sprites)

    # add interactable
    self.interactable_dict.update({'character': self.character})

    # mic handler
    #if self.mic is not None: self.interactable_dict.update({'input_mic_handler': InputMicHandler(objs=[self.interactable_dict['character']], mic=self.mic)})
    if self.mic is not None: self.interactable_dict['input_mic_handler'].objs.append(self.interactable_dict['character'])

    # handle this objects
    self.interactable_dict['input_key_handler'].objs.append(self.interactable_dict['character'])


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
    self.character.thing_sprites.add(self.thing)

    # determine position
    self.character.set_position(self.grid_world.grid_to_pos([10, 10]), is_init_pos=True)

    # add interactables
    self.interactable_dict.update({'win_canvas': CanvasWin(self.screen)})


  def define_game_logic(self):
    """
    define interactables
    """
    self.interactable_dict.update({'game_logic': ThingsGameLogic(self)})


  def win(self):
    """
    win condition for level
    """

    # deactivate henry
    self.character.is_active = False

    # activate win canvas
    self.interactable_dict['win_canvas'].enabled = True


  def reset(self):
    """
    reset level
    """

    # reset world
    self.grid_world.reset()
    self.character.reset()

    # add to sprites
    self.all_sprites.add(self.thing)
    self.character.thing_sprites.add(self.thing)

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
    self.character.set_position(self.grid_world.grid_to_pos([5, 20]), is_init_pos=True)
    self.thing.set_position(self.grid_world.grid_to_pos([22, 18]), is_init_pos=True)

    # add canvas
    self.interactable_dict.update({
      'level_text1': Text(self.screen, message='say "left", "right", etc. for movement', position=(50, 50), font_size='tiny_small', color=self.color_bag.text_level_background),
      'level_text2': Text(self.screen, message='say "go" to switch blocks', position=(50, 80), font_size='tiny_small', color=self.color_bag.text_level_background)
      })


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
    self.character.set_position(self.grid_world.grid_to_pos([22, 20]), is_init_pos=True)
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



class LevelHandler(Interactable):
  """
  simple level handler
  """

  def __init__(self, levels, start_level=0):

    # arguments
    self.levels = levels
    self.start_level = start_level

    # actual level id
    self.act_level = self.start_level

    # run loop flag
    self.run_loop = True


  def runs(self):
    """
    still running
    """
    return self.levels[self.act_level].runs() and self.run_loop


  def quit(self):
    """
    quitted game
    """
    return self.levels[self.act_level].interactable_dict['game_logic'].quit_game


  def event_update(self, event):
    """
    event update
    """
    self.levels[self.act_level].event_update(event)


  def update(self):
    """
    update
    """

    # check if level complete
    if self.levels[self.act_level].interactable_dict['game_logic'].complete:

      # update level
      self.act_level += 1

      # end game
      if self.act_level == len(self.levels): 
        self.run_loop = False
        self.act_level = 0

    # update
    self.levels[self.act_level].update() if self.run_loop else None



if __name__ == '__main__':
  """
  levels
  """
  
  import yaml

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))

  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(cfg['game']['screen_size'])

  # level creation
  levels = [Level_01(screen, cfg['game']['screen_size']), Level_02(screen, cfg['game']['screen_size']), LevelSquare(screen, cfg['game']['screen_size']), LevelMoveWalls(screen, cfg['game']['screen_size'])]

  # level handler
  level_handler = LevelHandler(levels=levels, start_level=0)

  # add clock
  clock = pygame.time.Clock()

  # game loop
  while level_handler.runs():
    for event in pygame.event.get():

      # input handling
      level_handler.event_update(event)

    # frame update
    level_handler.update()

    # update display
    pygame.display.flip()

    # reduce framerate
    clock.tick(cfg['game']['fps'])

  # end pygame
  pygame.quit()




  

