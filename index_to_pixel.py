#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"


#------------------------------------------------------------------------------

class Index2Pixel():
    """
    Utility class providing methods to convert indices to pixels and vice versa
    """

    def __init__(self, width: int):
        '''
        Constructor - sets width to the user-defined width
        '''
        self.width = width

    #--------------------------------------------------------------------------

    def get_width(self) -> int:
        '''
        Getter for attribute `width`
        '''
        return self.width

    def set_width(self, width: int) -> None:
        '''
        Setter for attribute `width`
        '''
        self.width = width

    #--------------------------------------------------------------------------

    def convert_pixel_to_index(self, pixel: tuple) -> int:
        '''
        Converts a pixel represented by a tuple to an index

                Parameters:
                        `pixel` (tuple): (row, col) pixel

                Returns:
                        `index` (int): index pixel
        '''
        row, col = pixel
        index = row * self.get_width() + col
        return index
    
    #--------------------------------------------------------------------------

    def convert_pixels_to_indices(self, pixels: set) -> set:
        '''
        Converts a set of pixels represented by tuples to a set of indices

                Parameters:
                        `pixels` (set): set of (row, col) pixels

                Returns:
                        `indices` (set): set of indices of pixels
        '''
        indices = set()
        for pixel in pixels:
            indices.add(self.convert_pixel_to_index(pixel))
        return indices

    #--------------------------------------------------------------------------

    def convert_index_to_pixel(self, index: int) -> tuple:
        '''
        Converts a pixel represented by an index to a tuple

                Parameters:
                        `index` (int): index of pixel

                Returns:
                        (tuple): (row, col) pixel
        '''
        row = index // self.get_width()
        column = index % self.get_width()
        return (row, column)

    #--------------------------------------------------------------------------

    def convert_indices_to_pixels(self, indices: set) -> set:
        '''
        Converts all pixels represented by a set of indices to a set of tuples

                Parameters:
                        `indices` (set): set of indices of pixels

                Returns:
                        `all_pixels` (set): set of (row, col) pixels
        '''
        all_pixels = set()
        for index in indices:
            pixel = self.convert_index_to_pixel(index)
            all_pixels.add(pixel)
        return all_pixels