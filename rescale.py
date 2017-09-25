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
    img = rescale(img, min_f, max_f, 3)
    labels = [rescale(l, min_f, max_f, 0) for l in labels]

    return img, labels

def rescale(I, fx, fy, order):
  for i, slc in I:
    I[i] = transform.rescale(slc, (fx,fy), order=order)

  return I
