"""
buttons
"""

import pygame
import pathlib

from glob import glob

from interactable import Interactable


class Button(Interactable):
  """
  Button base class
  """

  def __init__(self, surf, position=(0, 0), scale=(1, 1)):

    # arguments
    self.surf = surf
    self.position = position
    self.scale = scale

    # button images
    self.button_images = [pygame.Surface(size=(50, 25)), pygame.Surface(size=(50, 25))]

    # fills for standard button
    self.button_images[0].fill((100, 100, 0))
    self.button_images[1].fill((100, 150, 100))

    # button index
    self.button_index = 0

    # image
    self.image = self.button_images[self.button_index]
    self.rect = self.image.get_rect()


  def button_images_init(self):
    """
    init button images, e.g. loading and scaling
    """

    # rect for scaling
    self.rect = self.button_images[self.button_index].get_rect()

    # scale sprites
    self.button_images = [pygame.transform.scale(s, (self.rect.width * self.scale[0], self.rect.height * self.scale[1])) for s in self.button_images]

    # image and rect
    self.image = self.button_images[self.button_index]
    self.rect = self.image.get_rect()


  def button_press(self):
    """
    change button if pressed
    """
    
    # update index
    self.button_index = self.button_index + 1 if self.button_index < len(self.button_images) - 1 else 0

    # update image
    self.image = self.button_images[self.button_index]


  def draw(self):
    """
    draw button on surface
    """

    self.surf.blit(self.image, self.position)



class StartButton(Button):
  """
  start Button
  """

  def __init__(self, surf, position=(0, 0), scale=1):

    # init parents
    super().__init__(surf, position, scale)

    # root for images
    self.button_path = str(pathlib.Path(__file__).parent.absolute()) + "/art/buttons/"

    # change image
    self.button_images = [pygame.image.load(s).convert_alpha() for s in glob(self.button_path + 'start_button_*.png')]

    # init buttons
    self.button_images_init()



class EndButton(Button):
  """
  start Button
  """

  def __init__(self, surf, position=(0, 0), scale=1):

    # init parents
    super().__init__(surf, position, scale)

    # root for images
    self.button_path = str(pathlib.Path(__file__).parent.absolute()) + "/art/buttons/"

    # change image
    self.button_images = [pygame.image.load(s).convert_alpha() for s in glob(self.button_path + 'end_button_*.png')]

    # init buttons
    self.button_images_init()



class HelpButton(Button):
  """
  start Button
  """

  def __init__(self, surf, position=(0, 0), scale=1):

    # init parents
    super().__init__(surf, position, scale)

    # root for images
    self.button_path = str(pathlib.Path(__file__).parent.absolute()) + "/art/buttons/"

    # change image
    self.button_images = [pygame.image.load(s).convert_alpha() for s in glob(self.button_path + 'help_button_*.png')]

    # init buttons
    self.button_images_init()



if __name__ == '__main__':
  """
  button
  """

  import yaml

  from game_logic import GameLogic
  from color_bag import ColorBag
  
  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))

  # init pygame
  pygame.init()

  # colos
  color_bag = ColorBag()

  # init display
  screen = pygame.display.set_mode(cfg['game']['screen_size'])

  # button
  button_start = StartButton(screen, position=(100, 100), scale=(4, 4))
  button_end = EndButton(screen, position=(100, 200), scale=(4, 4))
  button_help = HelpButton(screen, position=(100, 300), scale=(4, 4))
  
  # press state
  button_start.button_press()

  # game logic
  game_logic = GameLogic()

  # add clock
  clock = pygame.time.Clock()

  # game loop
  while game_logic.run_loop:
    for event in pygame.event.get():

      # input handling
      game_logic.event_update(event)

    # frame update
    game_logic.update()

    # fill screen
    screen.fill(color_bag.background)

    # text update
    button_start.draw()
    button_end.draw()
    button_help.draw()

    # update display
    pygame.display.flip()

    # reduce frame rate
    clock.tick(cfg['game']['fps'])


  # end pygame
  pygame.quit()