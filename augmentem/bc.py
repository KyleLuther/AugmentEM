import numpy as np

def bc_augment(img, max_contrast=0.15, max_brightness=0.15):
  """Performs brightness and contrast augmentation on img.

    This consists of contrast, brightness distortion
    Refs/For more details, see:
            ELEKTRONN, https://elektronn.org
            DataProvider, https://github.com/torms3/DataProvider

  Args:
    img: (np array: <z,x,y,channel>) image to augment, pixel values in [0,1]
    contrast: (float) max amount to scale contrast
    brightness: (float) max amount to change brightness
  """
  contrast = 2*(np.random.rand() - 0.5)*max_contrast
  brightness = 2*(np.random.rand() - 0.5)*max_brightness
  img = bc(img, contrast, brightness)

  return img

def bc(img, contrast, brightness):
  img = img * (1 + contrast)
  img = img + brightness
  return img
