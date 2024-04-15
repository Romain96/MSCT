#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2024, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import argparse
import os

from msct_wrapper import MSCTWrapper
from msct_utils import load_image, convert_colour_to_grayscale, create_dir_remove_if_exist, normalize_image, test_if_image_is_colour
import time

#------------------------------------------------------------------------------

def process_image_msct(
        path: str, downsample: int, 
        enrich_area_factor: float, enrich_max_mser: float,
        mser_percent_height: float
    ):
    '''
    Builds a multi-scale component-tree (MSCT) on an image using a given number of subsampling and enrichment scales.
    Extracts single objects from enriched nodes.

            Parameters:
                    `path` (str): path to an image to process

                    `downsample` (int): number of downsampling steps (n° scales - 1) to use
                    `enrich_area_factor` (float): percentage of the total image area allowed fror enrighment process [0,1]
                    `enrich_max_mser` (float): maximum MSER value for node selection
                    `mser_percent_height` (float): percentage of the MSCT height to use for MSER computation [0,1]
    '''
    # loading the image
    image = load_image(path)
    is_colour = test_if_image_is_colour(image)
    gs_image = convert_colour_to_grayscale(image)
    max_area = int(gs_image.shape[0] * gs_image.shape[1] * enrich_area_factor)
    output_dir = 'output'
    create_dir_remove_if_exist(output_dir)

    reconstructed_dir = os.path.join(output_dir, 'reconstructed')
    downsampled_dir = os.path.join(output_dir, 'downsampled')
    msct_dir = os.path.join(output_dir, 'msct')
    fz_dir = os.path.join(output_dir, 'flatzones')
    create_dir_remove_if_exist(reconstructed_dir)
    create_dir_remove_if_exist(downsampled_dir)
    create_dir_remove_if_exist(msct_dir)
    create_dir_remove_if_exist(fz_dir)

    wrapper = MSCTWrapper()
    msct_times = []
    t = time.time()

    # Step 1 : subsampling steps and base component-tree building on the lowest scale image
    if is_colour:
        wrapper.downsample_image(
            image=gs_image, n=downsample, method='maximum', 
            use_median_filter=False, median_filter_size=3, 
            invert_image=True
        )
    else:
        wrapper.downsample_image(
            image=gs_image, n=downsample, method='maximum', 
            use_median_filter=False, median_filter_size=3, 
            invert_image=False
        )
    wrapper.save_downsampled_images(downsampled_dir=downsampled_dir, downsampled_name='f')
    msct_times.append(('sub', time.time() - t))
    t = time.time()
    wrapper.build_base_msct()
    msct_times.append(('base', time.time() - t))
    t = time.time()

    # computing MSER
    wrapper.compute_mser_percent_height(percent_height=mser_percent_height)
    msct_times.append(('mser', time.time() - t))
    t = time.time()

    # reconstructing the original image with the MSCT
    wrapper.reconstruct_image(reconstructed_dir=reconstructed_dir, reconstructed_name='reconstructed_0', normalize=False)

    # STEP 1 : enrichment steps
    for downsample_step in range(0, downsample):

        # augmenting MSCT by one scale
        t = time.time()
        wrapper.augment_msct_mser(max_area=max_area, max_mser=enrich_max_mser)
        msct_times.append((f"enrich_{downsample_step}", time.time() - t))
        t = time.time()    

        # reconstructing the original image with the MSCT
        wrapper.reconstruct_image(reconstructed_dir=reconstructed_dir, reconstructed_name=f"reconstructed_{downsample_step+1}", normalize=False)

        # computing MSER
        t = time.time()
        wrapper.compute_mser_percent_height(percent_height=mser_percent_height)
        msct_times.append((f"mser_{downsample_step}", time.time() - t))

    for txt, val in msct_times:
        print(f"{txt} : {val}")
    total_time = sum([i[1] for i in msct_times])
    print(f"MSCT building time : {total_time}")

#------------------------------------------------------------------------------

if __name__ == '__main__':

    # execute only if run as a script
    ap = argparse.ArgumentParser(prog="Multi-scale component-tree")
    ap.add_argument('image', help='Image to process (colour, grayscale, composite, multivalued...)')
    ap.add_argument('downsample', help='n° of downsampling steps')
    ap.add_argument('--mser_percent_height', help='Percentage of MSCT height used in MSER computation [0,1]', default=0.1)
    ap.add_argument('--enrich_area_factor', help='Maximum flat zone area to be enriched in percent of image size [0,1]', default=0.05)
    ap.add_argument('--enrich_max_mser', help='Maximum MSER value for candidate selection', default=1.0)

    args = vars(ap.parse_args())
    process_image_msct(
        path=args['image'], 
        downsample=int(args['downsample']), 
        enrich_area_factor=float(args['enrich_area_factor']),
        enrich_max_mser=float(args['enrich_max_mser']),
        mser_percent_height=float(args['mser_percent_height'])
    )