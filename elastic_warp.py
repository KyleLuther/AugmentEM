import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.interpolation import map_coordinates

def elastic_warp_augment(img, labels, n, max_sigma, d, clamp_borders=False):
  """Performs elastic deformation on img. Only

    Args:
      img: (np array: <z,y,x,ch>) image to augment
      labels: list of (np array: <z,y,x,ch>) labels of image
      n: number of grid points, must be >= 4
      max_sigma: max standard deviation of Gaussian filter used to blur image
      d: 2 means apply same warp to all slices, 3 means apply different warp
      clamp_borders: bool, if True does not distort image at borders
  """
  sigma = np.random.rand()*max_sigma
  if d == 2:
    img, labels = elastic_warp2d(img, labels, n, sigma, clamp_borders)
  elif d == 3:
    img, labels = elastic_warp3d(img, labels, n, sigma, clamp_borders)
  else:
    raise ValueError("Value of d not recognized: {}".format(d))

  return img, labels

def elastic_warp2d(img, labels, n, sigma, clamp_borders):
  assert(len(img.shape) == 4)
  z, y, x, ch = img.shape

  # Create coordinate mapping
  mapping = create_map(x,y,n)

  for i in range(z):
    # Apply coordinate mappings
    img[i] = apply_map(img[i], mapping, order=1)
    for l in labels:
      l[i] = apply_map(l, mapping, order=0)

  return img, labels

def elastic_warp3d(img, labels, n, sigma, clamp_borders):
  assert(len(img.shape) == 4)
  z, y, x, ch = img.shape

  for i in range(z):
    # Create coordinate mapping
    mapping = create_map(x,y,n)

    # Apply coordinate mappings
    img[i] = apply_map(img[i], mapping, order=1)
    for l in labels:
      l[i] = apply_map(l, mapping, order=0)

  return img, labels

def create_map(x,y,n):
  # Generate sparse grid points
  xs = np.linspace(0,x,n)
  ys = np.linspace(0,y,n)

  # Generate displacements on grid points
  dxs = (2*np.random.rand(n,n)-1)*sigma
  dys = (2*np.random.rand(n,n)-1)*sigma
  if clamp_borders:
    dxs[0,:] = dys[0,:] = 0.0
    dxs[:,0] = dys[:,0] = 0.0
    dxs[-1,:] = dys[-1,:] = 0.0
    dxs[:,-1] = dys[:,-1] = 0.0

  # Generate displacement splines
  x_spline = RectBivariateSpline(ys,xs,dxs)
  y_spline = RectBivariateSpline(ys,xs,dys)

  # Generate displacements on dense grid
  dX = x_spline(np.arange(y), np.arange(x), grid=True)
  dY = y_spline(np.arange(y), np.arange(x), grid=True)

  X, Y = np.meshgrid(np.arange(x), np.arange(y))

  # Generate coordinate mapping
  mapping = (Y+dY).flatten(), (X+dX).flatten()

  return mapping

def apply_map(arr, mapping, order):
  y,x,ch = arr.shape
  warped_arr = np.copy(arr)
  for c in range(ch):
    mapped = map_coordinates(arr[:,:,c], mapping, order=order)
    warped_arr[:,:,c] = np.reshape(mapped, (y,x))

  return warped_arr
