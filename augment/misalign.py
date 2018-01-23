import numpy as np

def misalign_augment(img, labels, p, delta, type, shift_labels):
  """Performs misalignment augmentation on img.

  Args:
    img: (np array: <z,y,x,ch>) image
    labels: (np array: <z,y,x,ch>) labeling of img
    p: probability to apply augmentation
    delta: max distance to misalign
    type: 'slip' or 'translation'
    shift_labels: if True, shift labels too (only applies for slip misalign)
  """
  if np.random.rand() < p:
    z = np.random.randint(0, img.shape[0])
    dx = np.random.randint(-delta, delta+1)
    dy = np.random.randint(-delta, delta+1)

    if type == 'slip':
      img, labels = apply_slip(img, labels, z, dx, dy, shift_labels)
    elif type == 'translation':
      img, labels = apply_translation(img, labels, z, dx, dy)
    else:
      raise ValueError('Unrecognized misalign type: {}'.format(type))

  return img, labels

def apply_slip(img, labels, z, dx, dy, shift_labels):
  _, y, x, _ = img.shape
  lx1, ly1 = max(dx, 0), max(dy, 0)
  lx2, ly2 = max(-dx, 0), max(-dy, 0)
  ux1, uy1 = min(x+dx,x), max(y+dy, 0)
  ux2, uy2 = min(x-dx,x), max(y-dy, 0)

  slip = np.zeros_like(img[z])
  slip[ly1:uy1,lx1:ux1] = img[z,ly2:uy2,lx2:ux2]
  img[z] = slip

  if shift_labels:
    for l in labels:
      slip = np.zeros_like(l[z])
      slip[ly1:uy1,lx1:ux1] = l[z,ly2:uy2,lx2:ux2]
      l[z] = slip

  return img, labels

def apply_translation(img, labels, z, dx, dy):
  _, y,x, _ = img.shape
  raise NotImplementedError
  ly = max(dy,0)
  uy = min(y+dy,y)
  lx = max(dx, 0)
  ux = min(x+dx,x)

  slip = np.zeros_like(img[z:])
  slip[:,ly:uy,lx:ux] = img[z:,ly:uy,lx:ux]
  img[z:] = slip

  for l in labels:
    slip = np.zeros_like(l[z:])
    slip[:,ly:uy,lx:ux] = l[z:,ly:uy,lx:ux]
    l[z:] = slip

  return img, labels
