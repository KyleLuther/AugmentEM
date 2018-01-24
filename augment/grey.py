import numpy as np

def grey_augment(img, max_contrast=0.15, max_brightness=0.15, max_gamma=1.0):
  """Performs grey value (histogram) augmentation on img.

    This consists of contrast, brightness, and gamma distortion
    Refs/For more details, see:
            ELEKTRONN, https://elektronn.org
            DataProvider, https://github.com/torms3/DataProvider

  Args:
    img: (np array: <z,x,y,channel>) image to augment, pixel values in [0,1]
    contrast: (float) max amount to scale contrast
    brightness: (float) max amount to change brightness
    gamma: (float) max amount of gamma to use
  """
  raise NotImplementedError
  constrast = 2*(np.random.rand() - 0.5)*max_contrast
  brightness = 2*(np.random.rand() - 0.5)*max_brightness
  gamma = 2*(np.random.rand() - 1)*max_gamma
  img = grey(img, contrast, brightness, gamma)

  return img

def grey(img, contrast, brightness, gamma):
  img = np.copy(img)
  img = img - np.min(img)
  img = img / np.max(img) if np.max(img) > 0.0 else img
  img *= 1 + contrast
  img += brightness
  img = np.clip(img, 0, 1)
  img **= 2.0**gamma

  return img
