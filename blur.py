import numpy as np
from scipy.ndimage.filters import gaussian_filter

def blur_augment(img, sigma, p):
  """Performs blur augmentation on img (simulate out of focus). Apply to
    each slice individually.

    Args:
      img: (np array: <z,y,x,channel>) image to augment
      sigma: max standard deviation of Gaussian filter used to blur image
      p: probability to perform blur
  """
  for i, slc in enumerate(img):
    if np.random.rand() < p:
      sigma = sigma*np.random.rand()
      img[i] = gaussian_filter(slc, sigma)

  return img
