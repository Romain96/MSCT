#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import cv2
import numpy as np
from enum import Enum

#------------------------------------------------------------------------------

class RCC8Relation(Enum):
    """
    Region Connection Calculus 8 - all possible connection types
    """
    INV = 0   # invalid connection
    DC = 1    # disconnected (not overlapping, no common border)
    EC = 2    # externally connected (common border, not overlapping)
    EQ = 3    # equal (a and b are the same exact pixels)
    PO = 4    # partially overlapping (subset of a and b is common)
    TPP = 5   # tangential proper part (a inside b with common border)
    TPPi = 6  # tangential proper part inverse (b inside a with common border)
    NTPP = 7  # non-tangential proper part (a fully inside b)
    NTPPi = 8 # non-tangential proper part inverse (b fully inside a)

#------------------------------------------------------------------------------

def rcc8_to_text(rel: RCC8Relation) -> str:
    '''
    Converts a RCC8 relation to a human-readable text

            Parameters:
                    `rel` (RCC8Relation): RCC8 enum

            Returns:
                    text (str): human-readable RCC8 relation
    '''
    text = ""
    if rel == RCC8Relation.INV:
        text = "invalid"
    elif rel == RCC8Relation.DC:
        text = "disconnected"
    elif rel == RCC8Relation.EC:
        text = "externally connected"
    elif rel == RCC8Relation.EQ:
        text = "equal"
    elif rel == RCC8Relation.PO:
        text = "partially overlapping"
    elif rel == RCC8Relation.TPP:
        text = "tangential proper part"
    elif rel == RCC8Relation.TPPi:
        text = "tangential proper part inverse"
    elif rel == RCC8Relation.NTPP:
        text = "non-tangential proper part"
    elif rel == RCC8Relation.NTPPi:
        text = "non-tangential proper part inverse"
    return text

#------------------------------------------------------------------------------

class RCC8():
    """
    This class provides a collection of methods to connect regions (set of pixels) together
    using Region Connection Calculus 8 or RCC8.
    """

    name = "Region Connection Calculus 8"

    def __init__(self):
        pass

    #--------------------------------------------------------------------------

    @staticmethod
    def compute_relation(p1: set, p2: set) -> tuple:
        '''
        Computes the RCC8 relation between two regions/sets of pixels

                Parameters:
                        `p1` (set): first region (set of pixels)
                        `p2` (set): second region (set of pixels)

                Returns:
                        tuple of 4 values :
                        (RCC8Relation): RCC8 relation between `p1` and `p2`
                        (RCC8Relation): RCC8 relation between `p2` and `p1`
                        (float): percentage of common pixels between `p1` and `p2` relative to `p1`
                        (float): percentage of common pixels between `p1` and `p2` relative to `p2`
        '''
        # empty sets produce an error
        if len(p1) == 0 or len(p2) == 0:
            return (RCC8Relation.INV, RCC8Relation.INV)
        
        # computing relation
        p1_boundary, p1_interior = RCC8.region_to_boundary_and_interior(p1)
        p2_boundary, p2_interior = RCC8.region_to_boundary_and_interior(p2)
        intersection_boundaries = RCC8.intersection(p1_boundary, p2_boundary)
        intersection_interiors = RCC8.intersection(p1_interior, p2_interior)
        percent_p1, percent_p2 = RCC8.get_percentage_common_pixels(p1, p2, intersection_boundaries, intersection_interiors)

        # p1 DC p2 iff intersection(p1, p2) = 0
        if len(intersection_boundaries) == 0 and len(intersection_interiors) == 0:
            return (RCC8Relation.DC, RCC8Relation.DC, percent_p1, percent_p2)
        
        # p1 EC p2 iff intersection(interior(p1), interior(p2)) = 0 and intersection(boundary(p1), boundary(p2)) != 0
        elif len(intersection_interiors) == 0 and len(intersection_boundaries) > 0:
            return (RCC8Relation.EC, RCC8Relation.EC, percent_p1, percent_p2)
        
        # p1 PO p2 iff intersection(interior(p1), interior(p2)) != 0 and there exists a pixel of interior(p1) not belonging to interior(p2) or other way around
        elif len(intersection_interiors) > 0 and len(RCC8.subset_p2_not_in_p1(p2_interior, p1_interior)) > 0:
            return (RCC8Relation.PO, RCC8Relation.PO, percent_p1, percent_p2)
        
        # p1 EQ p2 iff p1 = p2 (intersection(p1, p2) = p1 = p2)
        elif len(RCC8.intersection(p1, p2)) == len(p1) and len(p1) == len(p2):
            return (RCC8Relation.EQ, RCC8Relation.EQ, percent_p1, percent_p2)
        
        # p1 TTP p2 iff inclusion_ne(p1, p2) and intersection(boundary(p1), boundary(p2)) != 0
        elif RCC8.is_included_non_equal(p1, p2) and len(intersection_boundaries) > 0:
            return (RCC8Relation.TPP, RCC8Relation.TPPi, percent_p1, percent_p2)

        # p1 TTPi p2 iff inclusion_ne(p2, p1) and intersection(boundary(p1), boundary(p2)) != 0
        elif RCC8.is_included_non_equal(p2, p1) and len(intersection_boundaries) > 0:
            return (RCC8Relation.TPPi, RCC8Relation.TPP, percent_p1, percent_p2)

        # p1 NTTP p2 iff inclusion_ne(p1, p2) and intersection(boundary(p1), boundary(p2)) = 0
        elif RCC8.is_included_non_equal(p1, p2) and len(intersection_boundaries) == 0:
            return (RCC8Relation.NTPP, RCC8Relation.NTPPi, percent_p1, percent_p2)

        # p2 NTTPi p1 iff inclusion_ne(p1, p2) and intersection(boundary(p1), boundary(p2)) = 0
        elif RCC8.is_included_non_equal(p2, p1) and len(intersection_boundaries) == 0:
            return (RCC8Relation.NTPPi, RCC8Relation.NTPP, percent_p1, percent_p2)
        
        else:
            return (RCC8Relation.INV, RCC8Relation.INV, percent_p1, percent_p2)

    #--------------------------------------------------------------------------

    @staticmethod
    def region_to_boundary_and_interior(region: set) -> tuple:
        '''
        Computes the 4-neighbourhood boundary of a region and returns a subset of `pixels`
        forming the `boundary` and a subset of pixels forming the `interior` of the region.
        The union of `boundary` and `interior` is equal to 'region' and their intersection is empty.

                Parameters:
                        `region` (set): pixels forming a region

                Returns:
                        `boundary` (set): subset of pixels of `region` forming its boundary (4-neighbourhood)
                        `interior` (set): subset of pixels of `region` forming its interior
        '''

        rows = list()
        cols = list()
        for row, col in region:
            rows.append(row)
            cols.append(col)

        # find bounding box
        row_min = np.amin(rows, axis=0)
        row_max = np.amax(rows, axis=0)
        col_min = np.amin(cols, axis=0)
        col_max = np.amax(cols, axis=0)
        nrows = row_max - row_min + 1
        ncols = col_max - col_min + 1

        image = np.zeros((nrows, ncols), dtype=np.uint8)
        for index in range(0, len(rows)):
            image[rows[index] - row_min, cols[index] - col_min] = 255

        contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        cntr = contours[0]

        boundary_pixels = set()
        interior_pixels = set()
        for pixel in cntr:
            row = pixel[0,1]
            col = pixel[0,0]
            boundary_pixels.add((row + row_min, col + col_min))
        for pixel in region:
            if pixel not in boundary_pixels:
                interior_pixels.add(pixel)

        return boundary_pixels, interior_pixels
    
    #--------------------------------------------------------------------------

    @staticmethod
    def intersection(p1: set, p2: set) -> set:
        '''
        Intersection of sets of pixels.

                Parameters:
                        `p1` (set): set of pixels
                        `p2` (set): set of pixels

                Returns:
                    `inter` (set): set of common pixels of `p1` and `p2`
        '''
        inter = set()
        if len(p1) == 0 or len(p2) == 0:
            return inter
        for pixel in p1:
            if pixel in p2:
                inter.add(pixel)
        return inter
    
    #--------------------------------------------------------------------------

    @staticmethod
    def subset_p2_not_in_p1(p1: set, p2: set) -> set:
        '''
        Returns the subset of pixels of `p2` not belonging to `p1`.

                Parameters:
                        `p1` (set): set of pixels
                        `p2` (set): set of pixels

                Returns:
                        `p2_only` (set): subset of pixels of `p2` and not in `p1`
        '''
        p2_only = set()
        for pixel in p2:
            if pixel not in p1:
                p2_only.add(pixel)
        return p2_only

    #--------------------------------------------------------------------------

    @staticmethod
    def is_included_non_equal(p1: set, p2: set) -> bool:
        '''
        Checks whether the set `p1` is included but non equal to the set `p2`.

                Parameters:
                        `p1` (set): set of pixels
                        `p2` (set): set of pixels

                Returns:
                        (bool): True if `p1` is included but non equal to `p2`, False otherwise
        '''
        if len(p1) == 0 or len(p2) == 0:
            return False
        for pixel in p1:
            # all pixel of p1 should be in p2 for p1 to be included in p2
            if pixel not in p2:
                return False
        # if p2 has at most the same amount of pixels as p1 then we already checked that all pixels of p1 are in p2
        # so p2 does not have any pixel not belonging to p1 thus p1 is included and equal to p2
        # p2 needs to be stricly larger than p1 to have at least one pixel that does not belong to p1 and p2 simultaneously
        if len(p2) > len(p1):
            return True
        else:
            return False
        
    #--------------------------------------------------------------------------

    @staticmethod
    def get_percentage_common_pixels(p1: set, p2: set, inter_boundaries: set, inter_interiors: set):
        '''
        Computes the percentage of common pixels relative to p1 and to p2.

                Parameters:
                        `p1` (set): set of pixels P1
                        `p2` (set): set of pixels P2
                        `inter_boundaries` (set): subset of pixels of P1 inter P2 only for the boundaries
                        `inter_interiors` (set): subset of pixels of P1 inter P2 only for the interiors

                Returns:
                        `percent_p1` (float): percentage of common pixels of P1 and P2 relative to P1
                        `percent_p2` (float): percentage of common pixels of P1 and P2 relative to P2
        '''
        nb_common_pixels = len(inter_boundaries) + len(inter_interiors)
        nb_pixels_p1 = len(p1)
        nb_pixels_p2 = len(p2)
        percent_p1 = nb_common_pixels / nb_pixels_p1
        percent_p2 = nb_common_pixels / nb_pixels_p1
        return (percent_p1, percent_p2)
