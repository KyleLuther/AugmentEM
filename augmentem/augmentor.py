"""Provides data augmentation"""
import numpy as np

from .blur import blur_augment
from .box import box_augment
from .circle import circle_augment
from .crop import crop_augment
from .elastic_warp import elastic_warp_augment
from .flip import flip_augment
from .grey import grey_augment
from .misalign_slip import misalign_slip_augment
from .misalign_translation import misalign_translation_augment
from .missing_section import missing_section_augment
from .noise import noise_augment
from .rotate import rotate_augment
from .rotate90 import rotate90_augment
from .rescale import rescale_augment
from .sin import sin_augment

class Augmentor:
  def __init__(self, params):
    self.params = params
    self._init_params()

  def _init_params(self):
    augs = ['blur', 'box', 'circle', 'crop', 'elastic_warp', 'flip', 'grey',
            'misalign_slip', 'misalign_translation', 'missing_section',
            'noise', 'rotate', 'rotate90', 'rescale', 'sin']
    for aug in augs:
      if aug not in self.params.keys():
        self.params[aug] = False

  def __call__(self, img, labels=[]):
    return self.augment(img, labels)

  def augment(self, img, labels=[]):
    """Augments example.

    Args:
      img: (np array: <z,y,x,ch>) image
      labels: list of (int np array: <z,y,x,ch>), pixelwise labeling of image
      params: dict containing augmentation parameters, see code for details

    Returns:
      augmented img: image after augmentation
      augmented labels: labels after augmentation

    Note:
      augmented img,labels may not be same size as input img,labels
        because of warping
    """
    params = self.params
    img = np.copy(img)
    labels = [np.copy(l) for l in labels]

    # Crop
    if params['crop']:
      max_z = params['crop_z']
      max_r = params['crop_r']
      img, labels = crop_augment(img, labels, max_z, max_r)

    # Flip
    if params['flip']:
      img, labels = flip_augment(img, labels)

    # Rotate
    if params['rotate']:
      mode = params['rotate_mode']
      img, labels = rotate_augment(img, labels, mode)

    if params['rotate90']:
      img, labels = rotate90_augment(img, labels)

    # Rescale
    if params['rescale']:
      min_f = params['rescale_min']
      max_f = params['rescale_max']
      mode = params['rescale_mode']
      img, labels = rescale_augment(img, labels, min_f, max_f, mode)

    # Elastic warp
    if params['elastic_warp']:
      d = params['elastic_d']
      n = params['elastic_n']
      sigma = params['elastic_sigma']

      img, labels = elastic_warp_augment(img, labels, d, n, sigma)
      
    # Blur
    if params['blur']:
      sigma = params['blur_sigma']
      prob = params['blur_prob']
      img = blur_augment(img, sigma, prob)

    # Misalign slip
    if params['misalign_slip']:
      p = params['misalign_slip_prob']
      delta = params['misalign_slip_delta']
      shift_labels = params['misalign_slip_shift_labels']
      img, labels = misalign_slip_augment(img, labels, p, delta, shift_labels)

    # Misalign translation      
    if params['misalign_translation']:
      p = params['misalign_translation_prob']
      delta = params['misalign_translation_delta']
      shift_labels = params['misalign_translation_shift_labels']
      img, labels = misalign_translation_augment(img, labels, p, delta, shift_labels)

    # Missing Section
    if params['missing_section']:
      p = params['missing_section_prob']
      fill = params['missing_section_fill']
      img = missing_section_augment(img, p, fill)


    # Circle
    if params['circle']:
      p = params['circle_prob']
      r = params['circle_radius']
      fill = params['circle_fill']
      img = circle_augment(img, p, r, fill)

    if params['grey']:
      raise NotImplementedError
    
    if params['noise']:
      sigma = params['noise_sigma']
      img = noise_augment(img, sigma)

    if params['sin']:
      a = params['sin_a']
      f = params['sin_f']
      img = sin_augment(img, a, f)

    if params['box']:
      n = params['box_n']
      r = params['box_r']
      z = params['box_z']
      fill = params['box_fill']
      img = box_augment(img, n, r, z, fill)

    img = np.copy(img).astype(np.float32)
    labels = [np.copy(l) for l in labels]

    # Return
    if labels == []:
      return img
    else:
      return img, labels
