import numpy as np
from skimage import transform

def rotate_augment(img, labels):
  """Performs rotation augmentation on img. Rotate any angle in [0, 360).

  Args:
    img: (np array: <z,y,x,ch>) image
    labels: list of (np array: <z,y,x,ch>) labeling of img
  """
  z,y,x,ch = img.shape
  theta = np.random.randint(0, 360)
  img = rotate(img, theta, order=3)
  labels = [rotate(l, theta, order=0) for l in labels]

  return img, labels

def rotate(I, theta, order):
  z,x,y,ch = I.shape
  for i, slc in enumerate(I):
    for c in range(ch):
      img = slc[:,:,c]
      img = transform.rotate(img, theta, order=order,
              preserve_range=True, mode='constant').astype(slc.dtype)

      I[i,:,:,c] = img

  return I
