import numpy as np

def rotate90_augment(img, labels):
  """Performs rotation augmentation on img.
    Only do integer multiples of 90 deg. Only rotate about z-axis.

  Args:
    img: (np array: <z,y,x,ch>) image
    labels: list of (np array: <z,y,x,ch>), pixelwise labeling of img
  """
  rotations = (0,1,2,3)
  k = np.random.choice(rotations)
  rot_img = np.rot90(img,k=k, axes=(1,2))
  rot_labels = [np.rot90(l,k=k, axes=(1,2)) for l in labels]

  return rot_img, rot_labels
