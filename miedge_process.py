#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import argparse
import os
import numpy as np

from msct_utils import load_multi_channel_image
from miedge_csv_utils import find_dirs_in_dir
from msct_multi_channel_demo import get_image_and_annotations_from_dir, create_dir_remove_if_exist
from msct_multichannel_wrapper import MSCTMultiChannelWrapper
from msct_miedge_segmentation import save_channel_vectors_as_array, save_segmentation_as_label_image

#------------------------------------------------------------------------------

def apply_msct_with_vectors(input_dir: str) -> tuple[list[np.ndarray], list[list[int]]]:
    
    # params
    downsample_steps = 2# 1/16 -> 1/4 -> 1/1
    area_factor = 0.002
    mser_percent_height = 0.1
    enrich_max_mser = 1.0
    segmentation_max_mser = 1.0
    segmentation_max_area_factor = 0.0015
    segmentation_min_area = 100
    
    # get the data (image, annotations)
    print("\tloading the image")
    channel_images, annotations = get_image_and_annotations_from_dir(input_dir=input_dir)
    max_area = int(channel_images[0].shape[0] * channel_images[0].shape[1] * area_factor)

    wrapper = MSCTMultiChannelWrapper()
    wrapper.set_annotations(annotations)

    # Step 1 : subsampling steps and SSCT building on low scale image
    print("\tdowsampling")
    wrapper.downsample_image(
        channel_images=channel_images, base_channel=0, 
        n=downsample_steps, method='maximum', 
        use_median_filter=False, median_filter_size=3,
        invert_image=False, normalize=False
    )

    # base MSCT + MSER
    print("\tbase component-tree computation")
    wrapper.build_base_msct()
    wrapper.compute_mser_percent_height(percent_height=mser_percent_height)

    # STEP 2 : enrichment steps
    for i in range(0, downsample_steps):

        # augmenting MSCT by one scale + MSER
        print(f"\tenrichment at step {i}")
        wrapper.augment_msct_mser(max_area=max_area, max_mser=enrich_max_mser)
        wrapper.compute_mser_percent_height(percent_height=mser_percent_height)

    # STEP 3 : segmentation of custers into objects (cells/nuclei)
    print("\tsegmentation")
    nuclei, vectors = wrapper.divide_objects(
        max_mser=segmentation_max_mser, 
        max_area_factor=segmentation_max_area_factor,
        min_area=segmentation_min_area
    )

    return (wrapper.get_base_channel_image_at_scale(0), nuclei, vectors)

#------------------------------------------------------------------------------

def get_tif_dimensions(path: str) -> tuple[int, int]:
    channel_images = load_multi_channel_image(path=path)
    return (channel_images[0].shape[0], channel_images[0].shape[1])

#------------------------------------------------------------------------------

def process_all_images(input_dir: str, output_dir: str) -> None:
    '''
    Processes all images in the given directory with MSCT on DAPI and DTECMA + watershed segmentation 
    '''
    images_dir = find_dirs_in_dir(input_dir)
    for image_dir in images_dir:
        local_input_dir = os.path.join(input_dir, image_dir)
        local_output_dir = os.path.join(output_dir, image_dir)
        create_dir_remove_if_exist(local_output_dir)

        # apply MSCT + segmentation + vectors
        image, image_nuclei, image_vectors = apply_msct_with_vectors(input_dir=local_input_dir, )

        save_segmentation_as_label_image(data=image_nuclei, ref_image=image, output_dir=local_output_dir, output_name='labels')
        save_channel_vectors_as_array(data=image_vectors, output_dir=local_output_dir, output_name='vectors')

#------------------------------------------------------------------------------

if __name__ == "__main__":

    # execute only if run as a script
    ap = argparse.ArgumentParser(prog="MI*EDGE tumor classification")
    ap.add_argument('indir', help='input directory containing images in subdirectories')
    ap.add_argument('outdir', help="Output directory (will be erased and/or created)")

    args = vars(ap.parse_args())
    process_all_images(args['indir'], args['outdir'])