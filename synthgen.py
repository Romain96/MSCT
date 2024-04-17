#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import argparse
import numpy as np
from PIL import Image
import os
from scipy.ndimage import gaussian_filter
from skimage.draw import disk
from skimage.util import random_noise

#------------------------------------------------------------------------------

class Synthgen:

    dimensions = dict()
    dimensions[10000] = (100, 100)
    dimensions[20000] = (125, 160)
    dimensions[30000] = (150, 200)
    dimensions[40000] = (200, 200)
    dimensions[50000] = (200, 250)
    dimensions[60000] = (240, 250)
    dimensions[70000] = (250, 280)
    dimensions[80000] = (250, 320)
    dimensions[90000] = (300, 300)
    dimensions[100000] = (400, 250)
    dimensions[110000] = (400, 275)
    dimensions[120000] = (375, 320)
    dimensions[130000] = (400, 325)
    dimensions[140000] = (400, 350)
    dimensions[150000] = (400, 375)
    dimensions[160000] = (400, 400)
    dimensions[170000] = (425, 400)
    dimensions[180000] = (450, 400)
    dimensions[190000] = (475, 400)
    dimensions[200000] = (500, 400)
    dimensions[210000] = (500, 420)
    dimensions[220000] = (500, 440)
    dimensions[230000] = (500, 460)
    dimensions[240000] = (500, 480)
    dimensions[250000] = (500, 500)
    dimensions[260000] = (520, 500)
    dimensions[270000] = (540, 500)
    dimensions[280000] = (560, 500)
    dimensions[290000] = (580, 500)
    dimensions[300000] = (600, 500)
    dimensions[310000] = (620, 500)
    dimensions[320000] = (640, 500)
    dimensions[330000] = (660, 500)
    dimensions[340000] = (680, 500)
    dimensions[350000] = (700, 500)
    dimensions[360000] = (600, 600)
    dimensions[370000] = (740, 500)
    dimensions[380000] = (760, 500)
    dimensions[390000] = (780, 500)
    dimensions[400000] = (800, 500)
    dimensions[410000] = (656, 625)
    dimensions[420000] = (750, 560)
    dimensions[430000] = (688, 625)
    dimensions[440000] = (704, 625)
    dimensions[450000] = (720, 625)
    dimensions[460000] = (736, 625)
    dimensions[470000] = (752, 625)
    dimensions[480000] = (768, 625)
    dimensions[490000] = (700, 700)
    dimensions[500000] = (800, 625)

    #--------------------------------------------------------------------------

    @staticmethod
    def save_image(image: np.ndarray, output_dir: str, output_name: str) -> None:
        path = os.path.join(output_dir, f"{output_name}.png")
        img = Image.fromarray(image)
        img.save(path)

    #--------------------------------------------------------------------------

    @staticmethod
    def generate_random_coordinates(
        xmin: int, ymin: int, xmax: int, ymax: int, 
        radius: int, restrictions: tuple[int, int, int]
    ) -> tuple[int, int]:
        found = False
        niter = 0
        while not found or niter == 50:
            y = np.random.randint(ymin, ymax)
            x = np.random.randint(xmin, xmax)
            found = True
            # check if not too close from already existing disks centers
            for cy, cx, cradius in restrictions:
                if pow(x - cx, 2) + pow(y - cy, 2) <= pow(2 * max(radius, cradius), 2):
                    found = False
            if niter == 50:
                return (-1, -1)
            niter += 1
        return (y, x)

    #--------------------------------------------------------------------------

    @staticmethod
    def generate_synthetic_image(
        area: int, nb_cells: int, 
        radius: int, spread: float, 
        noise_variance: float, generate_masks: bool
    ) -> np.ndarray:
        '''
        Generates a synthetic cellular image containing `nb_cells` cells with radii between 
        `radius` x `spread` and `radius` x (1 + `spread`) and with Gaussian noise of variance = `noise_variance`

                Parameters:
                        `area` (int): pixel area of the desired image [10 000, 500 000] with increments of 10 000
                        `nb_cells` (int): desired number of cells to generate
                        `radius` (int): base radius of cells
                        `spread` (int): variation of radius for each cell (0 = all cell with radius = `radius`)
                        `noise_variance` (float): variance for the Gaussian noise addition
                        `generate_masks` (bool): whether to export ground truth masks

                Returns:
                        `final_image` (ndarray): generated synthetic image
                        `masks` (list): list of ground truth images if `masks` is True
        '''
        width, height = Synthgen.dimensions[area]

        # radii
        radius_min = round(radius - (radius * spread))
        radius_max = round(radius + (radius * spread))

        # addition to borders to avoir indices outside the image
        add = radius_max + 1

        # creating a base image
        image = np.zeros((height + 2 * add, width + 2 * add), dtype=np.uint8)

        # all possible radius values
        radii_values = [r for r in range(radius_min, radius_max + 1)]

        # choosing nb_cells number of cells with the correct probabilities
        radii = np.random.choice(radii_values, nb_cells)

        masks = []
        restrictions = []

        for radius in radii:

            # get a random coordinate with restrictions
            #y = np.random.randint(add, height - add)
            #x = np.random.randint(add, width - add)
            y, x = Synthgen.generate_random_coordinates(add, add, width - add, height - add, radius, restrictions)
            if (x < 0 and y < 0):
                break

            disk_coords = disk((y, x), radius=radius)
            restrictions.append((y, x, radius))
            val = np.random.randint(100, 255)
            image[disk_coords] = val
            # generating mground truth mask
            if generate_masks:
                mask = np.zeros_like(image, dtype=bool)
                mask[disk_coords] = True
                masks.append(mask)

        final_image = np.clip(image[add:height-add, add:width-add], a_min=0, a_max=255)
        final_image = random_noise(final_image, mode='gaussian', var=noise_variance)
        final_image = np.array(255 * final_image, dtype=np.uint8)
        # subtle smoothing
        final_image = gaussian_filter(final_image, sigma=0.5, radius=3)
        
        return final_image, masks
    
    #--------------------------------------------------------------------------

    @staticmethod
    def generate_synthetic_image_random(
        area: int, nb_cells: int, 
        radius: int, spread: float, 
        noise_variance: float
    ) -> np.ndarray:
        '''
        Generates a synthetic cellular image containing `nb_cells` cells with radii between 
        `radius` x `spread` and `radius` x (1 + `spread`) and with Gaussian noise of variance = `noise_variance`

                Parameters:
                        `area` (int): pixel area of the desired image [10 000, 500 000] with increments of 10 000
                        `nb_cells` (int): desired number of cells to generate
                        `radius` (int): base radius of cells
                        `spread` (int): variation of radius for each cell (0 = all cell with radius = `radius`)
                        `noise_variance` (float): variance for the Gaussian noise addition

                Returns:
                        `final_image` (ndarray): generated synthetic image
        '''
        width, height = Synthgen.dimensions[area]

        # radii
        radius_min = round(radius - (radius * spread))
        radius_max = round(radius + (radius * spread))

        # addition to borders to avoir indices outside the image
        add = radius_max + 1

        # creating a base image
        image = np.zeros((height + 2 * add, width + 2 * add), dtype=np.uint8)

        # all possible radius values
        radii_values = [r for r in range(radius_min, radius_max + 1)]

        # choosing nb_cells number of cells with the correct probabilities
        radii = np.random.choice(radii_values, nb_cells)

        for radius in radii:

            # get a random coordinate with restrictions
            y = np.random.randint(add, height - add)
            x = np.random.randint(add, width - add)

            disk_coords = disk((y, x), radius=radius)
            val = np.random.randint(100, 255)
            image[disk_coords] = val

        final_image = np.clip(image[add:height-add, add:width-add], a_min=0, a_max=255)
        final_image = random_noise(final_image, mode='gaussian', var=noise_variance)
        final_image = np.array(255 * final_image, dtype=np.uint8)
        # subtle smoothing
        final_image = gaussian_filter(final_image, sigma=0.5, radius=3)
        
        return final_image

#------------------------------------------------------------------------------

if __name__ == "__main__":

    # execute only if run as a script
    ap = argparse.ArgumentParser(prog="Synthetic cellular image generator")
    ap.add_argument('area', help='Desired pixel area')
    ap.add_argument('cells', help='Desired number of cells')
    ap.add_argument('radius', help='Base radius of cells')
    ap.add_argument('spread', help='Amount of radius variation in percent [0,1[')
    ap.add_argument('variance', help='Gaussian noise variance')
    ap.add_argument('--mask', help='Generate ground truth masks', action='store_true')
    ap.add_argument('--legacy', help='Full random generation without restriction nor masks', action='store_true')

    args = vars(ap.parse_args())
    user_area = int(args['area'])
    user_nb_cells = int(args['cells'])
    user_radius = int(args['radius'])
    user_spread = float(args['spread'])
    user_variance = float(args['variance'])

    if user_area < 0:
        user_area = 10000

    if user_nb_cells < 0:
        user_nb_cells = 0

    if user_radius < 0:
        user_radius = 1

    if user_spread < 0:
        user_spread = 0
    elif user_spread >= 1:
        user_spread = 0.95

    if user_variance < 0:
        user_variance = 0.0
    
    if args['legacy']:
        img = Synthgen.generate_synthetic_image_random(
            area=user_area, 
            nb_cells=user_nb_cells, 
            radius=user_radius, 
            spread=user_spread, 
            noise_variance=user_variance,
        )
        Synthgen.save_image(img, '.', 'synthgen')
    else:
        img, gts = Synthgen.generate_synthetic_image(
            area=user_area, 
            nb_cells=user_nb_cells, 
            radius=user_radius, 
            spread=user_spread, 
            noise_variance=user_variance,
            generate_masks=args['mask']
        )
        Synthgen.save_image(img, '.', 'synthgen')
        for i in range(0, len(gts)):
            Synthgen.save_image(gts[i], '.', f"synthgen_gt_{i}")
