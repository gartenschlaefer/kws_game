"""
screen capture for video presentation
"""

import pygame
import os

from common import create_folder, delete_png_in_path


class ScreenCapturer():
  """
  screen capture class for exporting videos
  """

  def __init__(self, screen, screen_size, fps, capture_path='./ignore/capture/', frame_path='frames/', frame_name='frame', external_cam=False):

    # params
    self.screen = screen
    self.screen_size = screen_size
    self.fps = fps

    # paths
    self.capture_path = capture_path
    self.frame_path = frame_path
    self.frame_name = frame_name

    # external cam
    if external_cam:
      self.cam = self.init_external_cam()

    # delete old data
    delete_png_in_path(self.capture_path + self.frame_path)

    # create folder for captured frames
    create_folder([self.capture_path + self.frame_path])

    # vars
    self.actual_frame_num = 0
    self.frame_container = []

    # downsample of fps
    self.downsample = 2
    self.downsample_count = 0


  def init_external_cam(self):
    """
    init external camera (not used)
    """

    import pygame.camera

    # init camera
    pygame.camera.init()

    # show cams
    print("cams: ", pygame.camera.list_cameras())

    # for external camera stuff
    cam = pygame.camera.Camera("/dev/video0", self.screen_size)

    # start camera
    cam.start()

    # use this to get images
    #screen_frame = self.cam.get_image()

    return cam


  def update(self):
    """
    update once per frame
    """

    # add image to container
    if self.downsample_count >= self.downsample:
      self.frame_container.append(pygame.image.tostring(self.screen, 'RGB'))
      self.downsample_count = 0

    # update frame number
    self.actual_frame_num += 1
    self.downsample_count += 1


  def save_video(self):
    """
    save as video format
    """

    # restore all images and save them
    for i, frame in enumerate(self.frame_container):

      # save image
      pygame.image.save(pygame.image.fromstring(frame, (self.screen_size[0], self.screen_size[1]), 'RGB'), '{}{}{}.png'.format(self.capture_path + self.frame_path, self.frame_name, i))

    # convert to video format
    os.system("avconv -r {} -i {}%d.png -s {}x{} -aspect 4:3 -y {}.avi".format(self.fps // self.downsample, self.capture_path + self.frame_path + self.frame_name, self.screen_size[0], self.screen_size[1], self.capture_path + 'out'))



if __name__ == '__main__':
  """
  capture
  """

  import soundfile

  # append paths
  import sys
  sys.path.append("./game")

  # game stuff
  from color_bag import ColorBag
  from game_logic import ThingsGameLogic
  from levels import Level_01, Level_02
  from classifier import Classifier
  from mic import Mic
  from text import Text


  # capture paths
  capture_path = './ignore/capture/'

  # --
  # mic (for sound capture)

  # params
  fs = 16000

  # window and hop size
  N, hop = int(0.025 * fs), int(0.010 * fs)

  # create classifier
  classifier = Classifier(file='./models/fstride_c-5.npz', verbose=False)

  # create mic instance
  mic = Mic(fs=fs, N=N, hop=hop, classifier=classifier, energy_thres=1e-4, device=8, is_audio_record=True)


  # --
  # game setup

  # fps
  fps = 60

  # size of display
  screen_size = width, height = 640, 480

  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(screen_size)


  # init screen capturer
  screen_capturer = ScreenCapturer(screen, screen_size, fps, capture_path=capture_path)


  # collection of game colors
  color_bag = ColorBag()
  text = Text(screen, color_bag)

  # level creation
  levels = [Level_01(screen, screen_size, color_bag, mic), Level_02(screen, screen_size, color_bag, mic)]

  # choose level
  level = levels[0]

  # game logic with dependencies
  game_logic = ThingsGameLogic(level, levels, text)

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
      level = game_logic.update()
      level.update()
      text.update()
      screen_capturer.update()


      # update display
      pygame.display.flip()

      # reduce framerate
      clock.tick(fps)


  # save video
  screen_capturer.save_video()

  # save audio
  soundfile.write(capture_path + 'out_audio.ogg', mic.collector.x_all, fs, format='ogg', subtype='vorbis')

  # end pygame
  pygame.quit()
