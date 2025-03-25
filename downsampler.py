#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import numpy as np

#------------------------------------------------------------------------------

class Downsampler:
	"""
	This class provides methods to downsample an image into a set of downsampled images of increasingly smaller sizes.

	Written by Romain PERRIN (<romain.perrin@unistra.fr>).
	"""

	def init(self):
		pass

	#--------------------------------------------------------------------------

	@staticmethod
	def downsample_image_minimum(image: np.ndarray) -> np.ndarray:
		"""
		Downsamples a grayscale image using a 2x2 window and retaining the minimum pixel value.

		Parameters
		----------
		image: ndarray
			Grayscale image to downsample (as Numpy array).

		Returns
		-------
		downsampled : ndarray
			Downsampled image (as Numpy array).

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""
	
		m, n = image.shape
		downsampled = image.reshape(m//2, 2, n//2, 2).min((1, 3))
		return downsampled

	#--------------------------------------------------------------------------

	@staticmethod
	def downsample_image_maximum(image: np.ndarray) -> np.ndarray:
		"""
		Downsamples a grayscale image using a 2x2 window and retaining the maximum pixel value.

		Parameters
		----------
		image : ndarray
			Grayscale image (as Numpy array).

		Returns
		-------
		downsampled : ndarray
			Downsampled image (as Numpy array).

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""
		m, n = image.shape
		downsampled = image.reshape(m//2, 2, n//2, 2).max((1, 3))
		return downsampled

	#--------------------------------------------------------------------------

	@staticmethod
	def downsample_image_mean(image: np.ndarray) -> np.ndarray:
		"""
		Downsamples a grayscale image using a 2x2 window and retaining the mean pixel value.

		Parameters
		----------
		image : ndarray
			Grayscale image (as Numpy array).

		Returns
		-------
		Downsampled : ndarray
			Downsampled image (as Numpy array).

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		m, n = image.shape
		downsampled = image.reshape(m//2, 2, n//2, 2).mean((1, 3))
		return downsampled.astype(np.uint8)

#------------------------------------------------------------------------------
