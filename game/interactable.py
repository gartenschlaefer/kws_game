"""
interactable interface class
"""

class Interactable():
  """
  Interactable interface class
  """

  def set_active(self, active):
    """
    set active
    """
    pass
    

  def speech_command(self, command):
    """
    speech command from mic
    """
    pass


  def is_moveable(self):
    """
    moveable flag
    """
    return False
    

  def direction_change(self, direction):
    """
    move direction
    """
    pass


  def action_key(self):
    """
    action key pressed
    """
    pass


  def enter_key(self):
    """
    enter key pressed
    """
    pass


  def esc_key(self):
    """
    esc key pressed
    """
    pass


  def r_key(self):
    """
    r key pressed
    """
    pass


  def reset(self):
    """
    reset
    """
    pass
    

  def event_update(self, event):
    """
    event update
    """
    pass


  def update(self):
    """
    update
    """
    pass


  def draw(self):
    """
    draw
    """
    pass