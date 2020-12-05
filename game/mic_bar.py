"""
mic class
"""

import pygame

from interactable import Interactable
from input_handler import InputKeyHandler


class MicBar(Interactable):
  """
  graphical bar for microphone energy measure
  """

  def __init__(self, mic, position, color_bag, size=(20, 40)):

    # init parents
    super().__init__()

    # mic
    self.mic = mic
    self.position = position
    self.color_bag = color_bag
    self.size = size

    # sprites group
    self.sprites = pygame.sprite.Group()

    # bar sprite
    self.bar_sprite = BarSprite(self.mic, position, size=self.size)

    # add to sprites
    self.sprites.add(self.bar_sprite)

    # input handler
    self.input_handler = InputKeyHandler(self)


  def action_key(self):
    """
    if action key is pressed
    """

    print("action")

    if self.bar_sprite.act_length > 5:
      self.bar_sprite.act_length -= 5


  def enter_key(self):
    """
    if enter key is pressed
    """
    
    print("enter")

    if self.bar_sprite.act_length < self.size[1] - 5:
      self.bar_sprite.act_length += 5


  def reset(self):
    """
    reset stuff
    """
    pass
    

  def event_update(self, event):
    """
    event for mic bar
    """

    # event handling
    self.input_handler.handle(event)


  def update(self):
    """
    update
    """
    pass



class BarSprite(pygame.sprite.Sprite):
  """
  wall class
  """

  def __init__(self, mic, position, color=(10, 200, 200), size=(20, 20)):

    # MRO check
    super().__init__()

    # vars
    self.mic = mic
    self.position = position
    self.color = color
    self.size = size

    # wall init
    self.image = pygame.surface.Surface(self.size)
    self.rect = self.image.get_rect()

    # set rect position
    self.rect.x, self.rect.y = self.position[0], self.position[1]

    # fill with color
    self.image.fill(self.color)

    # actual length
    self.act_length = 5


  def update(self):
    """
    update bar
    """

    # clear the image
    self.image.fill(self.color)

    # draw the rect
    pygame.draw.rect(self.image, (100, 0, 100), (5, self.size[1]-self.act_length, self.size[0] - 10, self.act_length))



if __name__ == '__main__':
  """
  mic bar
  """

  import yaml

  # append paths
  import sys
  sys.path.append("../")

  from classifier import Classifier
  from mic import Mic

  # game stuff
  from game_logic import GameLogic
  
  from levels import LevelMic
  #from text import Text


  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))


  # --
  # mic

  # create classifier
  classifier = Classifier(model_path='.' + cfg['classifier']['model_path'], verbose=cfg['classifier']['verbose'])

  # create mic instance
  mic = Mic(classifier=classifier, feature_params=cfg['feature_params'], mic_params=cfg['mic_params'], is_audio_record=cfg['game']['capture_enabled'])

  
  # --
  # game setup

  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(cfg['game']['screen_size'])

  # level creation
  levels = [LevelMic(screen, cfg['game']['screen_size'], mic)]

  # choose level
  level = levels[0]

  # game logic with dependencies
  game_logic = GameLogic()

  # add clock
  clock = pygame.time.Clock()

  # mic stream and update
  with mic.stream:

    # game loop
    while game_logic.run_loop:
      for event in pygame.event.get():

        # input handling
        game_logic.event_update(event)
        level.event_update(event)

      # frame update
      level.update()


      # #pygame.draw.rect(screen, (100, 0, 100), (5, 5, 5, 5))
      # meter_image = pygame.surface.Surface((5, 5))
      # meter_rect = meter_image.get_rect()

      # # set rect position
      # meter_rect.x, meter_rect.y = 0, 0

      # # fill with color
      # meter_image.fill((100, 0, 100))

      # screen.blit(meter_image, meter_rect)




      # update display
      pygame.display.flip()

      # reduce framerate
      clock.tick(cfg['game']['fps'])

  # end pygame
  pygame.quit()