import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.interpolation import map_coordinates

def elastic_warp_augment(img, labels, n, max_sigma, clamp_borders=True):
  """Performs elastic deformation on img. Warp all sections the same

    Args:
      img: (np array: <z,y,x,ch>) image to augment
      labels: list of (np array: <z,y,x,ch>) labels of image
      n: number of grid points, must be >= 4
      max_sigma: max distance to displace grid point
      clamp_borders: bool, if True does not distort image at borders
  """
  # generate sparse displacements
  dxs, dys = create_displacements(n, max_sigma, clamp_borders)

  # apply sparse displacements
  img, labels = elastic_warp(img, labels, dxs, dys)

  return img, labels

def elastic_warp(img, labels, dxs, dys):
  assert(len(img.shape) == 4)
  z, y, x, ch = img.shape

  # copy
  img = np.copy(img)
  labels = [np.copy(l) for l in labels]

  # generate dense coordinate mapping
  mapping = create_mapping(x, y, dxs, dys)

  # apply coordinate mapping to each section
  for i in range(z):
    img[i] = apply_mapping(img[i], mapping, order=1)
    for l in labels:
      l[i] = apply_mapping(l[i], mapping, order=0)

  return img, labels

def create_displacements(n, sigma, clamp_borders):
  dxs = (2*np.random.rand(n,n)-1)*sigma
  dys = (2*np.random.rand(n,n)-1)*sigma
  if clamp_borders:
    dxs[0,:] = dys[0,:] = 0.0
    dxs[:,0] = dys[:,0] = 0.0
    dxs[-1,:] = dys[-1,:] = 0.0
    dxs[:,-1] = dys[:,-1] = 0.0

  return dxs, dys

def create_mapping(x,y,dxs,dys):
  n = dxs.shape[0]

  # generate sparse grid points
  xs = np.linspace(0,x,n)
  ys = np.linspace(0,y,n)

  # generate displacement splines
  x_spline = RectBivariateSpline(ys,xs,dxs)
  y_spline = RectBivariateSpline(ys,xs,dys)

  # generate displacements on dense grid
  dX = x_spline(np.arange(y), np.arange(x), grid=True)
  dY = y_spline(np.arange(y), np.arange(x), grid=True)

  X, Y = np.meshgrid(np.arange(x), np.arange(y))

  # generate coordinate mapping
  mapping = (Y+dY).flatten(), (X+dX).flatten()

  return mapping

def apply_mapping(arr, mapping, order):
  y,x,ch = arr.shape
  warped_arr = np.copy(arr)
  for c in range(ch):
    mapped = map_coordinates(arr[:,:,c], mapping, order=order)
    warped_arr[:,:,c] = np.reshape(mapped, (y,x))

  return warped_arr
