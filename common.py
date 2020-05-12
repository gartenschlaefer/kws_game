"""
contains common fuctions
"""

import os

def create_folder(paths):
  """
  create folders in paths
  """

  # get all folder path to create
  for p in paths:

    # if it does not exist
    if not os.path.isdir(p):

      # create path
      os.makedirs(p)


if __name__ == '__main__':
  """
  main of common files
  """

  print("This is the common functions file.\n It includes for instance 'create_folder' ")