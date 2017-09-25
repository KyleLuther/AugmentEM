import numpy as np

def flip_augment(img, labels):
  """Performs flip augmentation on img.

  Args:
    img: (np array: <z,y,x,ch>) image
    labels: (np array: <z,y,x,ch>) labeling of img
  """
  # z flip
  if np.random.rand() < 0.5:
    img = np.flip(img, axis=0)
    labels = [np.flip(l, axis=0) for l in labels]

  # x flip
  if np.random.rand() < 0.5:
    img = np.flip(img, axis=1)
    labels = [np.flip(l, axis=1) for l in labels]

  return img, labels
