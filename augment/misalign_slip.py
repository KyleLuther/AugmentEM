import numpy as np

def misalign_slip_augment(img, labels, p, delta, shift_labels):
  """Performs misalignment augmentation on img.

  Args:
    img: (np array: <z,y,x,ch>) image
    labels: list of [(np array: <z,y,x,ch>)] labeling of img
    p: probability to apply augmentation
    delta: max distance to misalign
    shift_labels: if True, shift labels too
  """
  if np.random.rand() < p:
    dz = np.random.randint(0, img.shape[0])
    dy = np.random.randint(-delta, delta+1)
    dx = np.random.randint(-delta, delta+1)

    img, labels = misalign_slip(img, labels, dz, dy, dx, shift_labels)

  return img, labels

def misalign_slip(img, labels, dz, dy, dx, shift_labels):
  img[dz] = _misalign_slip(img[dz], dy, dx)

  if shift_labels:
    for l in labels:
      l[dz] = _misalign_slip(l[dz], dy, dx)

  return img, labels

def _misalign_slip(arr, dy, dx):
  y,x,ch = arr.shape
  lx1, ly1 = max(dx, 0), max(dy, 0)
  lx2, ly2 = max(-dx, 0), max(-dy, 0)
  ux1, uy1 = min(x+dx,x), max(y+dy, 0)
  ux2, uy2 = min(x-dx,x), max(y-dy, 0)

  slip = np.zeros_like(arr)
  slip[ly1:uy1,lx1:ux1] = arr[ly2:uy2,lx2:ux2]

  return slip
