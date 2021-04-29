"""
contains common fuctions
"""

import os


def check_folders_existance(folders, empty_check=False):
  """
  check if all folders exist
  """

  for f in folders:

    # existance check
    if not os.path.isdir(f):
      print("folder does not exists: ", f)
      return False

    # empty check
    if empty_check:
      if not len(os.listdir(f)):
        print("folder is empty: ", f)
        return False

  return True


def check_files_existance(files):
  """
  check if file exist
  """

  for file in files:

    # none type exception
    if file is None: continue
    
    # check file existance
    if not os.path.isfile(file): return False

  return True


def create_folder(paths):
  """
  create folders in paths
  """

  # get all folder path to create
  for p in paths:

    # path exists check
    if not os.path.isdir(p): os.makedirs(p)


def delete_files_in_path(paths, file_ext='.png'):
  """
  delete png files in folder
  """

  for p in paths:
    if os.path.isdir(p):
      for file in os.listdir(p):
        if file.endswith(file_ext):
          print("delete file: ", p + file)
          os.remove(p + file)


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

  print("\nThis is the common functions file.\nIt includes for instance 'create_folder'\n")