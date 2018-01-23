import numpy as np

def flip_augment(img, labels):
  """Performs flip augmentation on img.

  Args:
    img: (np array: <z,y,x,ch>) image
    labels: (np array: <z,y,x,ch>) labeling of img
  """
  # z flip
  if np.random.rand() < 0.5:
    img, labels = flip(img, labels, 0)

  # x flip
  if np.random.rand() < 0.5:
    img, labels = flip(img, labels, 1)

  return img, labels

def flip(img, labels, axis):
  img = np.flip(img, axis=axis)
  labels = [np.flip(l, axis=axis) for l in labels]

  return img, labels
