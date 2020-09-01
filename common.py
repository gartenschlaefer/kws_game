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


def delete_png_in_path(path):
  """
  delete png files in folder
  """

  if os.path.isdir(path):
    for file in os.listdir(path):
      if file.endswith('.png'):
        os.remove(path + file)


def s_to_hms_str(x):
  """
  convert seconds to reasonable time format
  """

  m, s = divmod(x, 60)
  h, m = divmod(m, 60)

  return '[{:02d}:{:02d}:{:02d}]'.format(int(h), int(m), int(s))


if __name__ == '__main__':
  """
  main of common files
  """

  print("This is the common functions file.\n It includes for instance 'create_folder' ")