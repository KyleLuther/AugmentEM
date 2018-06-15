import numpy as np

def missing_section_augment(img, p, fill):
  """Performs misalignment augmentation on img.

  Args:
    img: (np array: <z,y,x,ch>) image
    p: probability to apply augmentation
    fill: 'zero' or 'noise'
  """
  for z in range(img.shape[0]):
    if np.random.rand() < p:
      img[z] = missing_section(img[z], fill)

  return img

def missing_section(img, fill):
  if fill == 'zero':
    img = np.zeros_like(img)
  elif fill == 'noise':
    sigma = np.std(img)
    mu = np.mean(img)
    img = mu + sigma*np.sqrt(12)*(np.random.rand(*(img.shape))-0.5)

  return img
