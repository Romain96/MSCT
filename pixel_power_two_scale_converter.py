#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

#------------------------------------------------------------------------------

class PixelPower2ScaleConverter():
    """
    Utility class providing methods to upscale or downscale indices of pixels with power of 2 scales
    """

    def area_to_global_scale(self, local_area: int, local_scale: int) -> int:
        '''
        Return the pixel area at a given scale (scale k, pixels divided by 2^k) by 
        successive subsampling by a 2x2 window converted at the global scale 
        (scale 0, pixel divided by 2^0=1).
        From each scale k to a scale k-1, one pixel yields 4 new ones

                Parameters:
                        local_area (int): pixel area at the node scale (local)
                        local_scale (int): node scale

                Returns:
                        (int) pixel area at the global scale
        '''
        return int(local_area * (4**local_scale))

    #--------------------------------------------------------------------------

    def to_upper_scale(self, pixel: tuple[int, int]) -> set[tuple[int, int]]:
        '''
        Convert the given pixel to an upper scale (x2) yielding 4 new pixels

                Parameters:
                        pixel (tuple): (row, col) pixel 

                Returns:
                        (set): set of pixels at upper scale
        '''
        row, column = pixel
        p0 = (int(2 * row), int(2 * column))
        p1 = (int(2 * row), int(2 * column + 1))
        p2 = (int(2 * row + 1), int(2 * column))
        p3 = (int(2 * row + 1), int(2 * column + 1))
        return set([p0, p1, p2, p3])

    #--------------------------------------------------------------------------

    def to_lower_scale(self, pixel: tuple[int, int]) -> set[tuple[int, int]]:
        '''
        Convert the given pixel to a lower scale (/2) yielding one new pixel (4 -> 1)

                Parameters:
                        pixel (tuple): (row, col) pixel

                Returns:
                        (set): set of pixels at lower scale
        '''
        row = pixel[0]
        column = pixel[1]
        p0 = (int(row // 2), int(column // 2))
        return set([p0])

    #--------------------------------------------------------------------------

    def to_n_upper_scale(self, pixel: tuple[int, int], n: int) -> set[tuple[int, int]]:
        '''
        Convert the given pixel to n upper scales (/2)

                Parameters:
                        pixel (tuple): (row, col) pixel

                Returns:
                        (set): set of pixels at n upper scale
        '''
        final_upscaled_pixels = set()
        intermediate_pixels = set([pixel])
        for _ in range(0, n):
            tmp = set()
            for inter_pixel in intermediate_pixels:
                tmp.update(self.to_upper_scale(inter_pixel))
            intermediate_pixels = tmp
        final_upscaled_pixels.update(intermediate_pixels)
        return final_upscaled_pixels

    #--------------------------------------------------------------------------

    def to_n_lower_scale(self, pixel: tuple[int, int], n: int) -> set[tuple[int, int]]:
        '''
        Convert the given pixel to n lower scale (/2) yielding one new pixel (4 -> 1)

                Parameters:
                        pixel (tuple): (row, col) pixel

                Returns:
                        (set): set of pixels at n lower scale
        '''
        final_downscaled_pixels = set()
        intermediate_pixels = set([pixel])
        for _ in range(0, n):
            tmp = set()
            for inter_pixel in intermediate_pixels:
                tmp.update(self.to_lower_scale(inter_pixel))
            intermediate_pixels = tmp
        final_downscaled_pixels.update(intermediate_pixels)
        return final_downscaled_pixels

    #--------------------------------------------------------------------------

    def convert_pixels_to_upper_scale(self, pixels: set[tuple[int, int]]) -> set[tuple[int, int]]:
        '''
        Convert all pixels to an upper scale (x2) each pixel yields 4 new pixels

                Parameters:
                        pixels (set): (row, col) pixels

                Returns:
                        all_upscaled_pixels (set): set of upscaled pixels
        '''
        all_upscaled_pixels = set()
        for pixel in pixels:
            upscaled_pixels = self.to_upper_scale(pixel)
            all_upscaled_pixels.update(upscaled_pixels)
        return all_upscaled_pixels

    #--------------------------------------------------------------------------

    def convert_pixels_to_lower_scale(self, pixels: set[tuple[int, int]]) -> set[tuple[int, int]]:
        '''
        Convert all pixels to a lower scale (/2) each group of 4 pixels yields one new pixel

                Parameters:
                        pixels (set): (row, col) pixels

                Returns:
                        all_downscaled_pixels (set): set of pixels at lower scale
        '''
        all_downscaled_pixels = set()
        for pixel in pixels:
            downscaled_pixels = self.to_lower_scale(pixel)
            all_downscaled_pixels.update(downscaled_pixels)
        return all_downscaled_pixels

    #--------------------------------------------------------------------------

    def convert_pixels_to_n_upper_scale(self, pixels: set[tuple[int, int]], n: int) -> set[tuple[int, int]]:
        '''
        Convert all pixels to an upper scale (x2) n times each pixel yields 4 new pixels per scale

                Parameters:
                        pixels (set): (row, col) pixels

                Returns:
                        all_upscaled_pixels (set): set of pixels at upper scale
        '''
        all_upscaled_pixels = set()
        for pixel in pixels:
            upscaled_pixels = self.to_n_upper_scale(pixel, n)
            all_upscaled_pixels.update(upscaled_pixels)
        return all_upscaled_pixels

    #--------------------------------------------------------------------------

    def convert_pixels_to_n_lower_scale(self, pixels: set[tuple[int, int]], n: int) -> set[tuple[int, int]]:
        '''
        Convert all pixels to a lower scale (/2) n times each group of 4 pixels yields one new pixel per scale

                Parameters:
                        pixels (set): (row, col) pixels

                Returns:
                        all_downscaled_pixels (set): set of pixels at lower scale
        '''
        all_downscaled_pixels = set()
        for pixel in pixels:
            downscaled_pixels = self.to_n_lower_scale(pixel, n)
            all_downscaled_pixels.update(downscaled_pixels)
        return all_downscaled_pixels
