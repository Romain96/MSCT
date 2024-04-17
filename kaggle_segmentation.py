#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import argparse
import os
import csv
import numpy as np

from msct_wrapper import MSCTWrapper
from intersection_over_union import evaluate_image
from msct_utils import create_dir_remove_if_exist, load_image, test_if_image_is_colour, convert_colour_to_grayscale

#------------------------------------------------------------------------------

def get_subdirectories(path: str) -> list:
    return [f.name for f in os.scandir(path) if f.is_dir()]

#------------------------------------------------------------------------------

def apply_msct_method(image_path: str, image_name: str, output_dir: str):

    # default parameters
    subsample = 2

    enrich_area_factor = 0.10
    enrich_max_mser = 1.0
    mser_percent_height = 0.10 # 10% of the MSCT height
    segmentation_max_mser = 1.0 # max MSER when filtering the simplified MSERTree
    segmentation_min_area = 16 # unused

    # loading the image and converting to grayscale
    image = load_image(image_path)
    is_colour = test_if_image_is_colour(image)
    gs_image = convert_colour_to_grayscale(image)
    max_area = int(gs_image.shape[0] * gs_image.shape[1] * enrich_area_factor)

    output_dir = os.path.join(output_dir, image_name)
    create_dir_remove_if_exist(output_dir)

    reconstructed_dir = os.path.join(output_dir, 'reconstructed')
    objects_dir = os.path.join(output_dir, 'objects')
    create_dir_remove_if_exist(reconstructed_dir)
    create_dir_remove_if_exist(objects_dir)

    wrapper = MSCTWrapper()
    
    # preprocess
    if is_colour:
        wrapper.subsample_image(
            image=gs_image, 
            n=subsample, 
            method='maximum', 
            use_median_filter=True, 
            median_filter_size=3, 
            invert_image=True
        )
    else:
        wrapper.subsample_image(
            image=gs_image, 
            n=subsample, 
            method='maximum', 
            use_median_filter=True, 
            median_filter_size=3, 
            invert_image=False
        )

    # building Gk
    wrapper.build_base_msct()
    wrapper.compute_mser_percent_height(percent_height=mser_percent_height)
    wrapper.reconstruct_image(reconstructed_dir=reconstructed_dir, reconstructed_name='rec_0', normalize=False)

    # enriching from Gk to G0
    for subsample_step in range(0, subsample):
        wrapper.augment_msct_mser(max_area=max_area, max_mser=enrich_max_mser)
        wrapper.compute_mser_percent_height(percent_height=mser_percent_height)
        wrapper.reconstruct_image(reconstructed_dir=reconstructed_dir, reconstructed_name=f"rec_{subsample_step+1}", normalize=False)

    # segmentation
    objects = wrapper.divide_objects(max_mser=segmentation_max_mser, min_area=segmentation_min_area)
    wrapper.export_objects_as_images(images=objects, output_dir=objects_dir, all=True)
    #kaggle = wrapper.export_objects_for_kaggle(objects=objects, debug=False)

#------------------------------------------------------------------------------

def process_kaggle(input_dir: str, output_dir: str) -> None:
    subdirs = get_subdirectories(input_dir)
    image_index = 1
    for subdir in subdirs:
        print(f"Processing image {image_index}/{len(subdirs)} : {subdir}")
        input_image_path = os.path.join(input_dir, subdir, 'images', f"{subdir}.png")
        apply_msct_method(image_path=input_image_path, image_name=subdir, output_dir=output_dir)          
        image_index += 1

#------------------------------------------------------------------------------

if __name__ == '__main__':

    # execute only if run as a script
    ap = argparse.ArgumentParser(prog="MSCT Kaggle segmentation")
    ap.add_argument('indir', help='Input directory (Kaggle Bowl dataset)')
    ap.add_argument('outdir', help='Output directory')
    ap.add_argument('--debug', help='Prints all debug informations (log)', action='store_true')

    args = vars(ap.parse_args())
    stats = process_kaggle(input_dir=args['indir'], output_dir=args['outdir'])