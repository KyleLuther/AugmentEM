import numpy as np
from skimage import draw

def circle_augment(img, p, max_r):
  """Performs circle augmentation on img (simulate dirt and tears).
      Zeros out circular region of image

  Args:
    img: (np array: <z,y,x,ch>) image
    p: probability to perform augmentation
    max_r: max radius of circle
  """
  #raise NotImplementedError
  for z in range(img.shape[0]):
    if np.random.rand() < p:
      y, x, ch = img[z].shape
      r = np.random.rand() * max_r
      yc, xc = np.random.randint(y), np.random.randint(x)
      img = circle(img[z], yc, xc, r)

  return img

def circle(img, yc, xc, r):
  y, x, ch = img.shape
  yy, xx = draw.circle(yc, xc, r, shape=(y,x))
  img[yy, xx, :] = 0.0

  return img
