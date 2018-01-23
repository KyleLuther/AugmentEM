import numpy as np

def sin_augment(img, max_a, max_f):
  """Performs sinusoidal augmentations on img.
      Adds sinusoidally varying levels to image

    Args:
      img: (np array: <z,y,x,channel>) image to augment
      max_a: max amplitude of sinusoid
      max_f: max wavelen = 1 / min freq
  """
  for z in range(img.shape[0]):
    a = max_a * np.random.rand()
    f = max_f * np.random.rand()
    phi = 2 * np.pi * np.random.rand()
    theta = 2 * np.pi * np.random.rand()

    img[z] = sin(img[z], a, f, phi, theta)

  return img

def sin(img, a, f, phi, theta):
  y,x,_ = img.shape
  yy,xx = np.mgrid[:y,:x]
  zz = np.sin(theta) * yy + np.cos(theta) * xx
  ss = np.expand_dims(np.sin(2 * np.pi * f * zz + phi), -1)
  #return img + a * np.sin(2 * np.pi * f * np.arange(img.shape[0])+phi).reshape((img.shape[0],1))
  return img + a * ss
