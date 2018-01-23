import numpy as np
from scipy.ndimage.filters import gaussian_filter

def noise_augment(img, max_sigma):
  """Adds random gaussian noise to image.

    Args:
      img: (np array: <z,y,x,channel>) image to augment
      sigma: max standard deviation of Gaussian filter used to blur image
  """
  sigma = max_sigma*np.random.rand()
  img = noise(img, sigma)

  return img

def noise(img, sigma):
  return img + np.random.normal(scale=sigma, size=img.shape)
