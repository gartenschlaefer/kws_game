"""
mic class
"""

import pygame
import numpy as np

from interactable import Interactable
from input_handler import InputKeyHandler
from color_bag import ColorBag
from text import Text


class MicBar(Interactable):
  """
  graphical bar for microphone energy measure
  """

  def __init__(self, surf, mic, position, bar_size=(20, 40), scale_margin=(10, 5), energy_frame_update=4, border=2, tick_length=10, min_db=-80):

    # mic
    self.surf = surf
    self.mic = mic
    self.position = position
    self.bar_size = bar_size
    self.scale_margin = scale_margin
    self.energy_frame_update = energy_frame_update
    self.border = border
    self.tick_length = tick_length
    self.min_db = min_db

    # sprites group
    #self.sprites = pygame.sprite.Group()

    # bar sprite
    #self.bar_sprite = BarSprite(self.mic, position, bar_size=self.bar_size, scale_margin=self.scale_margin, min_db=self.min_db)

    # colors
    self.color_bag = ColorBag()

    # input handler
    self.input_handler = InputKeyHandler(self)

    # image size
    self.image_size = (self.bar_size[0] + self.scale_margin[0], self.bar_size[1] + self.scale_margin[1] * 2,)

    # bar init
    self.image = pygame.surface.Surface(self.image_size)
    self.rect = self.image.get_rect()

    # set rectangle position
    self.rect.x, self.rect.y = self.position[0], self.position[1]

    # lengths
    self.act_length, self.total_length = 5, self.bar_size[1]

    # energy thresh
    self.energy_thresh_position = int(self.mic.mic_params['energy_thresh_db'] * (self.bar_size[1] / self.min_db))

    # draw background
    self.draw_background()


  def reload_thresh(self):
    """
    reload thresh from mic
    """
    self.change_energy_thresh_db_pos(self.mic.mic_params['energy_thresh_db'])


  def change_energy_thresh_db_pos(self, e):
    """
    energy threshold change
    """

    # energy thresh
    self.energy_thresh_position = int(e * (self.bar_size[1] / self.min_db))

    # draw background again
    self.draw_background()


  def action_key(self):
    """
    if action key is pressed
    """

    print("action")

    if self.act_length > 5: self.act_length -= 5


  def enter_key(self):
    """
    if enter key is pressed
    """
    
    print("enter")

    if self.act_length < self.bar_size[1] - 5: self.act_length += 5


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
    
    # debug
    #self.update()
    #return

    # read mic
    self.mic.read_mic_data()

    # frame update
    if len(self.mic.collector.e_all) > self.energy_frame_update:

      # mean energy of frames
      e_mu = np.mean(self.mic.collector.e_all)

      # reset collection
      self.mic.collector.reset_collection_all()

      # set bar accordingly
      self.act_length = np.clip((10 * np.log10(e_mu) / (-1 * self.min_db) + 1) * self.total_length, a_min=0, a_max=self.total_length)

      # update bar
      self.draw_meter()


  def draw(self):
    """
    draw
    """
    self.surf.blit(self.image, self.position)


  def draw_background(self):
    """
    draw background
    """

    # fill with background color
    self.image.fill(self.color_bag.mic_bar_background)

    # border
    pygame.draw.rect(self.image, self.color_bag.mic_bar_border, (0, self.scale_margin[1] - self.border, self.bar_size[0] + 2 * self.border, self.total_length + 2 * self.border))

    # scale
    scale_ticks = [i * self.total_length // (np.abs(self.min_db) // 10) for i in range(np.abs(self.min_db) // 10 + 1)]
    scale_tick_names = [i * -10 for i in range(len(scale_ticks))]

    # draw scale ticks
    [pygame.draw.rect(self.image, self.color_bag.mic_bar_meter_tick, (self.bar_size[0], self.scale_margin[1] + i, self.tick_length, 2)) for i in scale_ticks]

    # draw text
    texts = [Text(self.image, message=str(m), position=(self.border + self.bar_size[0] + self.tick_length + 2, self.scale_margin[1] - 6 + i), font_size='tiny', color=self.color_bag.mic_bar_meter_tick) for i, m in zip(scale_ticks, scale_tick_names)]
    texts.append(Text(self.image, message='[db]', position=(self.border + self.bar_size[0] + self.tick_length + 2, self.scale_margin[1] // 2 - 6), font_size='tiny', color=self.color_bag.mic_bar_meter_tick))
    [text.draw() for text in texts]

    # draw meter background
    self.draw_meter()


  def draw_meter(self):
    """
    update bar by drawing the rectangle
    """

    # draw meter background
    pygame.draw.rect(self.image, self.color_bag.mic_bar_meter_background, (self.border, self.scale_margin[1], self.bar_size[0], self.total_length))
    
    # draw meter bar
    pygame.draw.rect(self.image, self.color_bag.mic_bar_meter, (5 + self.border, self.total_length - self.act_length + self.scale_margin[1], self.bar_size[0] - 10, self.act_length))

    # energy thresh
    pygame.draw.rect(self.image, self.color_bag.mic_bar_energy_thresh, (self.border, self.scale_margin[1] + self.energy_thresh_position, self.bar_size[0] + self.tick_length + 2, 2))



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
  from game_logic import GameLogic
  from levels import LevelMic


  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))


  # --
  # mic

  # create classifier
  classifier = Classifier(cfg_classifier=cfg['classifier'], root_path='../')
  
  # create mic instance
  mic = Mic(classifier=classifier, mic_params=cfg['mic_params'], is_audio_record=True)

  
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

  # init stream
  mic.init_stream()

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

      # update display
      pygame.display.flip()

      # reduce framerate
      clock.tick(cfg['game']['fps'])

  # end pygame
  pygame.quit()