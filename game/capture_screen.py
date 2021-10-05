"""
screen capture for video presentation
"""

import pygame
import os
import soundfile

from datetime import datetime
from interactable import Interactable
from input_handler import InputKeyHandler

# append paths
import sys
sys.path.append("../")
from common import create_folder, delete_files_in_path


class ScreenCapturer(Interactable):
  """
  screen capture class for recording pygame screens
  """

  def __init__(self, screen, cfg_game, frame_name='frame', root_path='./'):

    # argumentsscreenshot_name
    self.screen = screen
    self.cfg_game = cfg_game
    self.frame_name = frame_name
    self.root_path = root_path

    # vars
    self.screenshot_name = 'screenshot_{}-'.format(datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))

    # input handler
    self.input_handler = InputKeyHandler(objs=[self])

    # shortcuts
    self.screen_size = cfg_game['screen_size']
    self.fps = cfg_game['fps']

    # paths
    self.paths = dict((k, self.root_path + v) for k, v in self.cfg_game['paths'].items())

    # delete old data
    delete_files_in_path([self.paths['frame_path']], file_ext='.png')

    # create folder for captured frames
    create_folder(list(self.paths.values()))

    # vars
    self.actual_frame_num = 0
    self.frame_container = []
    self.screenshot_container = []

    # downsample of fps
    self.downsample = 2
    self.downsample_count = 0


  def event_update(self, event):
    """
    event update
    """
    self.input_handler.event_update(event)


  def r_key(self):
    """
    r key pressed
    """
    print("screenshot")
    self.screenshot_container.append(pygame.image.tostring(self.screen, 'RGB'))


  def update(self):
    """
    update once per frame
    """

    # return if deactivated
    if not self.cfg_game['capture_enabled']: return

    # add image to container
    if self.downsample_count >= self.downsample:
      self.frame_container.append(pygame.image.tostring(self.screen, 'RGB'))
      self.downsample_count = 0

    # update frame number
    self.actual_frame_num += 1
    self.downsample_count += 1


  def save_video(self, mic=None):
    """
    save as video format
    """

    # save screenshots
    [pygame.image.save(pygame.image.fromstring(frame, (self.screen_size[0], self.screen_size[1]), 'RGB'), '{}{}{}.png'.format(self.paths['screenshot_path'], self.screenshot_name, i)) for i, frame in enumerate(self.screenshot_container)]

    # return if deactivated
    if not self.cfg_game['capture_enabled']: return

    # save frames
    [pygame.image.save(pygame.image.fromstring(frame, (self.screen_size[0], self.screen_size[1]), 'RGB'), '{}{}{}.png'.format(self.paths['frame_path'], self.frame_name, i)) for i, frame in enumerate(self.frame_container)]

    # save audio
    if mic is not None: soundfile.write('{}out_audio.wav'.format(self.paths['capture_path']), mic.collector.x_all, mic.feature_params['fs'], subtype=None, endian=None, format=None, closefd=True)

    # convert to video format
    try:
      os.system("ffmpeg -framerate {} -start_number 0 -i {}%d.png -i {}out_audio.wav -vcodec mpeg4 {}.avi".format(self.fps // self.downsample, self.paths['frame_path'] + self.frame_name, self.paths['capture_path'], self.paths['capture_path'] + 'out'))
    except:
      print("***Problem with conversions of frames to video")


if __name__ == '__main__':
  """
  capture
  """

  import yaml
  
  # game stuff
  from game_logic import ThingsGameLogic
  from levels import Level_01, Level_02, LevelHandler
  from classifier import Classifier
  from mic import Mic


  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))

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

  # init screen capturer
  screen_capturer = ScreenCapturer(screen, cfg['game'], root_path='./')

  # level creation
  levels = [Level_01(screen, cfg['game']['screen_size'], mic)]

  # level handler
  level_handler = LevelHandler(levels=levels, start_level=0)

  # add clock
  clock = pygame.time.Clock()

  # init mic stream
  #mic.init_stream()

  # mic stream and update
  with mic.stream:

    # game loop
    while level_handler.runs():
      for event in pygame.event.get():

        # input handling
        level_handler.event_update(event)
        screen_capturer.event_update(event)

      # frame update
      level_handler.update()
      screen_capturer.update()

      # update display
      pygame.display.flip()

      # reduce framerate
      clock.tick(cfg['game']['fps'])


  # save video
  screen_capturer.save_video(mic)

  # end pygame
  pygame.quit()
