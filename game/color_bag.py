"""
color bag class
"""

class ColorBag():
  """
  Input Handler class
  """

  def __init__(self):

    # background
    self.background = (255, 255, 255)

    # ordinary walls
    self.wall = (10, 200, 200)

    # moving walls
    self.active_move_wall = (200, 100, 100)
    self.default_move_wall = (10, 100, 100)

    # text color
    self.text_win = (50, 100, 100)
    self.text_menu = (50, 100, 100)
    self.text_menu_active = (100, 50, 75)

    # mic bar
    #self.mic_bar_meter = (210, 100, 20)
    self.mic_bar_background = (255, 100, 255)
    self.mic_bar_border = (0, 0, 0)
    self.mic_bar_energy_thresh = (210, 100, 20)
    self.mic_bar_meter = (100, 0, 100)
    self.mic_bar_meter_tick = (0, 0, 0)
    self.mic_bar_meter_background = (10, 200, 200)

    # canvas
    self.canvas_background = (230, 210, 200)
    self.canvas_win_backgound = (230, 210, 200, 128)
    self.canvas_option_background = (255, 255, 255)



