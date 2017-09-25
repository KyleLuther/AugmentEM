import numpy as np
from skimage import transform

def rescale_augment(img, labels, min_f, max_f):
  """Performs flip augmentation on img. (Only rescale in xy)

  Args:
    img: (np array: <z,y,x,ch>) image
    labels: (np array: <z,y,x,ch>) labeling of img
    min_f: min rescale factor
    max_f: max rescale factor
  """
  z,y,x,ch = img.shape
  fx = min_f + np.random.rand()*(max_f-min_f)
  fy = min_f + np.random.rand()*(max_f-min_f)
  img = rescale(img, fx, fy, 3)
  labels = [rescale(l, fx, fy, 0) for l in labels]

  return img, labels

def rescale(I, fx, fy, order):
  z,x,y,ch = I.shape
  for i, slc in enumerate(I):
    for c in range(ch):
      img = slc[:,:,c]
      # Scale
      img = transform.rescale(img, (fx,fy), order=order,
              preserve_range=True, mode='constant').astype(slc.dtype)

      # Pad
      Y = max(0,y-img.shape[0])
      X = max(0,x-img.shape[1])
      img = np.pad(img, ((Y//2,Y-Y//2),(X//2,X-X//2)),mode='constant')

      # Crop
      Y = img.shape[0] - y
      X = img.shape[1] - x
      img = img[Y//2:Y//2+y,X//2:X//2+x]

      I[i,:,:,c] = img

  return I
