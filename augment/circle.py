import numpy as np
from skimage.draw import circle

def circle_augment(img, p, max_r):
  """Performs circle augmentation on img (simulate dirt).
      Zeros out circular region of image

    Args:
      img: (np array: <z,y,x,channel>) image to augment
      p: probability to perform augmentation
      max_r: max radius of circle
  """
  if np.random.rand() < p:
    z, y, x, ch = img.shape
    r = np.random.rand() * max_r
    zc, yc, xc = np.random.randint(z), np.random.randint(y), np.random.randint(x)
    yy, xx = circle(yc, xc, r, shape=(y,x))

    img[zc, yy, xx, :] = 0.0

  return img
