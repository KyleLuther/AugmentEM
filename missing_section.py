import numpy as np

def missing_section_augment(img, p):
  """Performs misalignment augmentation on img.

  Args:
    img: (np array: <z,y,x,ch>) image
    p: probability to apply augmentation
  """
  if np.random.rand() < p:
    z = np.random.randint(0, img.shape[0])
    img[z] = np.zeros_like(img[z])

  return img, labels
