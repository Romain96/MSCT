#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import numpy as np

from multi_scale_component_tree_node import MultiScaleComponentTreeNode

#------------------------------------------------------------------------------

class Otsu():
    """
    Provides a collection of methods to compute the Otsu's criteria
    """

    @staticmethod
    def compute_otsu_criteria(image: np.ndarray, threshold: int) -> float:
        '''
        Computes the Otsu's criteria on the given image for the given threshold value.

                Parameters:
                        `image` (ndarray): a grayscale image
                        `threshold` (int): a threshold value in [0,255]

                Returns:
                        (float): the Otsu's criteria of the image `image` for the threshold value `threshold`
        '''
        if threshold < 0:
            threshold = 0
        elif threshold > 255:
            threshold = 255

        thresholded_image = np.zeros(image.shape, dtype=np.uint8)
        thresholded_image[image >= threshold] = 1

        # computing weights
        n_pixels_image = image.size
        n_pixels_thresh = np.count_nonzero(thresholded_image)
        w1 = n_pixels_thresh / n_pixels_image
        w0 = 1 - w1

        # returning infinity if one of the class is empty
        if w1 == 0 or w0 == 0:
            return np.inf

        # finding all pixels belonging to each class (2)
        values_pixels_0 = image[thresholded_image == 0]
        values_pixels_1 = image[thresholded_image == 1]

        # computing the variance of each class
        variance_0 = np.var(values_pixels_0) if len(values_pixels_0) > 0 else 0
        variance_1 = np.var(values_pixels_1) if len(values_pixels_1) > 0 else 0

        return w0 * variance_0 + w1 * variance_1

    #--------------------------------------------------------------------------

    @staticmethod
    def compute_optimal_threshold(image: np.ndarray) -> int:
        '''
        Computes the optimal threshold value i.e. the one minimizing the Otsu criteria
        of the given image.

                Parameters:
                        `image` (ndarray): a grayscale image

                Returns:
                        optimal (int): the optimal threshold according to the Otsu's criteria in [0,255]
        '''
        threshold_range = range(0, np.max(image) + 1)
        otsu_criterias = [Otsu.compute_otsu_criteria(image, threshold) for threshold in threshold_range]
        optimal = threshold_range[np.argmin(otsu_criterias)]
        return optimal

    #--------------------------------------------------------------------------
    
    @staticmethod
    def compute_otsu_criteria_msct(root: MultiScaleComponentTreeNode, node: MultiScaleComponentTreeNode) -> float:
        '''
        Compute the Otsu criteria of a multi-scale component-tree node.

                Parameters:
                        `root` (MultiScaleComponentTreeNode): root node of the MSCT
                        `node` (MultiScaleComponentTreeNode): node for which the Otsu's criteria is to be computed

                Returns:
                        otsu (float): the Otsu's criteria of the node `node`
        '''
        background_values = Otsu.__msct_otsu_gather_background_class(root, node)
        foreground_values = Otsu.__msct_otsu_gather_foreground_class(node)

        n_pixels_image = len(background_values) + len(foreground_values)
        n_pixels_thresh = len(foreground_values)
        w1 = n_pixels_thresh / n_pixels_image
        w0 = 1 - w1

        # returning infinity if one of the class is empty
        if w1 == 0 or w0 == 0:
            return np.inf

        # computing the variance of each class
        variance_0 = np.var(background_values) if len(background_values) > 0 else 0
        variance_1 = np.var(foreground_values) if len(foreground_values) > 0 else 0

        otsu = w0 * variance_0 + w1 * variance_1
        return otsu

    #--------------------------------------------------------------------------

    @staticmethod
    def __msct_otsu_gather_background_class(root: MultiScaleComponentTreeNode, node: MultiScaleComponentTreeNode) -> list:
        '''
        Gathers the background class of a given node i.e. all pixels not belonging to the subtree of `node`.

                Parameters:
                        `root` (MultiScaleComponentTreeNode): root node of the MSCT
                        `node` (MultiScaleComponentTreeNode): node being the root of the foreground subtree

                Returns:
                        values (list): list of pixels forming the background class
        '''
        values = []
        to_process = []
        to_process.append(root)

        while len(to_process) > 0:
            local_node = to_process.pop(0)
            if local_node != node:
                local_values = [local_node.get_level()] * local_node.get_global_area()
                values += local_values
                for child in local_node.get_children():
                    to_process.append(child)
        
        return values
    
    #--------------------------------------------------------------------------

    @staticmethod
    def __msct_otsu_gather_foreground_class(node: MultiScaleComponentTreeNode) -> list:
        '''
        Gathers the foreground class of a given node i.e. all pixels belonging to the subtree of `node`.

                Parameters:
                        `node` (MultiScaleComponentTreeNode): node being the root of the foreground subtree

                Returns:
                        values (list): list of pixels forming the foreground class
        '''
        values = []
        to_process = []
        to_process.append(node)

        while len(to_process) > 0:
            local_node = to_process.pop(0)
            local_values = [local_node.get_level()] * local_node.get_global_area()
            values += local_values
            for child in local_node.get_children():
                to_process.append(child)

        return values


