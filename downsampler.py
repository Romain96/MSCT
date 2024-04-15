#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import numpy as np

#------------------------------------------------------------------------------

class Downsampler:

    def init(self):
        pass

    #--------------------------------------------------------------------------

    @staticmethod
    def downsample_image_minimum(image: np.ndarray) -> np.ndarray:
        '''
        Downsampling using a 2x2 window and keeping the min value

                Parameters:
                        `image` (ndarray): Numpy array of the image

                Returns:
                        `downsampled` (ndarray): Numpy array of the subsampled image
        '''
        m, n = image.shape
        downsampled = image.reshape(m//2, 2, n//2, 2).min((1, 3))
        return downsampled

    #--------------------------------------------------------------------------

    @staticmethod
    def downsample_image_maximum(image: np.ndarray) -> np.ndarray:
        '''
        Downsampling using a 2x2 window and keeping the max value

                Parameters:
                        `image` (ndarray): Numpy array of the image

                Returns:
                        `downsampled` (ndarray): Numpy array of the subsampled image
        '''
        m, n = image.shape
        downsampled = image.reshape(m//2, 2, n//2, 2).max((1, 3))
        return downsampled

    #--------------------------------------------------------------------------

    @staticmethod
    def downsample_image_mean(image: np.ndarray) -> np.ndarray:
        '''
        Downsampling using a 2x2 window and keeping the mean value

                Parameters:
                        `image` (ndarray): Numpy array of the image

                Returns:
                        `Downsampled` (ndarray): Numpy array of the subsampled image
        '''
        m, n = image.shape
        downsampled = image.reshape(m//2, 2, n//2, 2).mean((1, 3))
        return downsampled.astype(np.uint8)

#------------------------------------------------------------------------------
