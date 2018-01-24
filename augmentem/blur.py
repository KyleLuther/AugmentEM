import numpy as np
from scipy.ndimage.filters import gaussian_filter

def blur_augment(img, max_sigma, p):
  """Apply blur to sections (simulate out of focus)

  Args:
    img: (np array: <z,y,x,ch>) image 
    max_sigma: max standard deviation of Gaussian filter used to blur image
    p: probability to perform blur
  """
  for i in range(img.shape[0]):
    if np.random.rand() < p:
      sigma = max_sigma*np.random.rand()
      img[i] = blur(img[i], sigma)

  return img

def blur(img, sigma):
  return gaussian_filter(img, sigma)
