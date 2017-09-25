"""Provides data augmentation"""
import numpy as np

from flip import flip_augment
from rotate90 import rotate90_augment
from blur import blur_augment
from elastic_warp import elastic_warp_augment
from misalign import misalign_augment
from missing_section import missing_section_augment
from rescale import rescale_augment
from circle import circle_augment

class Augmentor:
  def __init__(self, params):
    self.params = params
    self._init_params()

  def _init_params(self):
    augs = ['elastic_warp', 'flip', 'rot90', 'blur',
            'misalign', 'missing_section', 'rescale', 'circle']
    for aug in augs:
      if aug not in self.params.keys():
        self.params[aug] = False

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

    # Elastic warp
    if params['elastic_warp']:
      n = params['elastic_n_grid_point']
      max_sigma = params['elastic_max_sigma']
      warp_d = params['elastic_d']

      img, labels = elastic_warp_augment(img, labels, n,
                                        max_sigma, warp_d, clamp_borders=True)

    # Flip
    if params['flip']:
      img, labels = flip_augment(img, labels)

    # Rotate
    if params['rot90']:
      img, labels = rotate90_augment(img, labels)

    # Blur
    if params['blur']:
      sigma = params['blur_sigma'] #3
      prob = params['blur_prob'] #0.01
      img = blur_augment(img, sigma, prob)

    # Misalign
    if params['misalign']:
      p = params['misalign_prob']
      delta = params['misalign_delta']
      type = params['misalign_type']
      shift_labels = params['misalign_label_shift']
      img, labels = misalign_augment(img, labels, p, delta, type, shift_labels)

    # Missing Section
    if params['missing_section']:
      p = params['missing_section_prob']
      img = missing_section_augment(img, p)

    # Rescale
    if params['rescale']:
      min_f = params['rescale_min']
      max_f = params['rescale_max']
      img, labels = rescale_augment(img, labels, min_f, max_f)

    # Circle
    if params['circle']:
      p = params['circle_prob']
      r = params['circle_radius']
      img = circle_augment(img, p, r)

    # Return
    if labels == []:
      return img
    else:
      return img, labels
