"""
resize images to desired resolution
"""

import cv2
from glob import glob

if __name__ == '__main__':
  """
  main function
  """

  # paths
  #img_path = './ignore/data/misc/'
  #img_path = './ignore/data/shovelnaut/'
  img_path = './ignore/data/ice_monster/'
  out_path = './ignore/resized_img/'

  # resize size (for 16x16 pixelart)
  rs_size = (16*20, 16*20)

  # get image files
  image_files = glob(img_path + '*.png')

  for i, image_file in enumerate(image_files):

    file_name = image_file.split('/')[-1]
    print("img: ", image_file)
    print("file_name: ", file_name)

    # read image
    img = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

    # interpolation for pixel art
    rs_img = cv2.resize(img, rs_size, interpolation=cv2.INTER_NEAREST)
    #rs_img = cv2.resize(img, rs_size, interpolation=cv2.INTER_AREA)

    # interpolation for other images
    #rs_img = cv2.resize(img, rs_size, interpolation=cv2.INTER_LINEAR)
    #rs_img = cv2.resize(img, rs_size, interpolation=cv2.INTER_CUBIC)
    #rs_img = cv2.resize(img, rs_size, interpolation=cv2.INTER_LANCZOS4)

    # output file name
    out_file_name = out_path + 'rs-' + file_name

    # save image
    cv2.imwrite(out_file_name, rs_img)

