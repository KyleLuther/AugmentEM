import numpy as np

def box_augment(img, max_n, max_r, max_z, fill_list):
  """Fills random rectangular regions of image with zeros or noise

  Args:
    img: (np array: <z,y,x,ch>) image
    max_n: max number of boxes
    max_r: max size of box in xy
    max_z: max size of box in z
    fill_list: list of possible fill options: 'zero', 'mean', 'min', max', 'noise'
  """
  n = np.random.randint(max_n)
  for i in range(n):
    z, y, x, ch = img.shape
    z0, y0, x0 = np.random.randint(z), np.random.randint(y), np.random.randint(x)
    dz, dy, dx = np.random.randint(1, max_z+1), np.random.randint(1, max_r+1), np.random.randint(1, max_r+1)
    fill = np.random.choice(fill_list)
    
    img = box(img, (z0, y0, x0), (z0+dz, y0+dy, x0+dx), fill)

  return img

def box(img, c0, c1, fill):
  z0, y0, x0 = c0
  z1, y1, x1 = c1
  region = img[z0:z1,y0:y1,x0:x1]
  if fill == 'zero':
    region[...] = 0.0
  elif fill == 'mean':
    region[...] = np.mean(img)
  elif fill == 'min':
    region[...] = np.min(img)
  elif fill == 'max':
    region[...] = np.max(img)
  elif fill == 'noise':
    sigma = np.std(img)
    mu = np.mean(img)
    region[...] = mu+sigma*np.sqrt(12)*(np.random.rand(*region.shape)-0.5)
  else:
    raise ValueError('fill value, {}, not recognized'.format(fill))

  return img
