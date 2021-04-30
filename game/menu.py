"""
pygame menu
"""

import pygame
import pygame_menu

if __name__ == '__main__':
  """
  main
  """
  import yaml

  from game_logic import GameLogic

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))


  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(cfg['game']['screen_size'])

  # game logic with dependencies
  game_logic = GameLogic()

  # add clock
  clock = pygame.time.Clock()

  menu = pygame_menu.Menu(300, 400, 'Welcome', theme=pygame_menu.themes.THEME_BLUE)

  menu.add.text_input('Name: ', default='chris')
  menu.add.selector('diff: ', [('hard', 1), ('easy', 2)])
  menu.add.button('play', print("play"))
  menu.add.button('Quit', pygame_menu.events.EXIT)
  menu.mainloop(screen)

  # game loop
  while game_logic.run_loop:
    for event in pygame.event.get():

      # input handling
      game_logic.event_update(event)

    # update display
    pygame.display.flip()

    # reduce framerate
    clock.tick(cfg['game']['fps'])

  # end pygame
  pygame.quit()