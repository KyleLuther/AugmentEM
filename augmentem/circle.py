import numpy as np
from skimage import draw

def circle_augment(img, p, max_r, fill_list):
  """Performs circle augmentation on img (simulate dirt and tears).
      Zeros out circular region of image

  Args:
    img: (np array: <z,y,x,ch>) image
    p: probability to perform augmentation
    max_r: max radius of circle
    fill_list: options to fill circle
  """
  #raise NotImplementedError
  for z in range(img.shape[0]):
    if np.random.rand() < p:
      y, x, ch = img[z].shape
      r = np.random.rand() * max_r
      yc, xc = np.random.randint(y), np.random.randint(x)
      fill = np.random.choice(fill_list)
      
      img[z] = circle(img[z], yc, xc, r, fill)

  return img

def circle(img, yc, xc, r, fill):
  y, x, ch = img.shape
  yy, xx = draw.circle(yc, xc, r, shape=(y,x))

  if fill == 'zero':
    img[yy, xx, :] = 0.0
  elif fill == 'mean':
    img[yy, xx, :] = np.mean(img)
  elif fill == 'min':
    img[yy, xx, :] = np.min(img)
  elif fill == 'max':
    img[yy, xx, :] = np.max(img)
  elif fill == 'noise':
    sigma = np.std(img)
    mu = np.mean(img)
    img[yy, xx, :] = mu+sigma*np.sqrt(12)*(np.random.rand(*region.shape)-0.5)
  else:
    raise ValueError('fill value, {}, not recognized'.format(fill))

  return img
