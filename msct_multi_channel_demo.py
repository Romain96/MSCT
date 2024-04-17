#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import argparse
import os
import numpy as np

from msct_multichannel_wrapper import MSCTMultiChannelWrapper
from msct_utils import load_multi_channel_image, create_dir_remove_if_exist
from miedge_csv_utils import read_single_image_annotations_csv_as_rc
import time

#------------------------------------------------------------------------------

def get_image_and_annotations_from_dir(input_dir: str) -> tuple[list[np.ndarray], dict]:
    '''
    Returns the TIF image and the loaded annotations from the CSV file inside the given directory.

            Parameters :
                    `input_dir` (str): input directory path

            Returns :
                `image` (ndarray): loaded multi channel TIF image
                `annotations` (dict): loaded dictionnary from CSV dot annotations
    '''
    image_path = os.path.join(input_dir, 'image.tif')
    annotations_path = os.path.join(input_dir, 'annotations.csv')

    if not os.path.exists(image_path):
        Exception(f"Image {image_path} does not exist !")
    if not os.path.exists(annotations_path):
        Exception(f"Annotations {annotations_path} does not exist !")

    image = load_multi_channel_image(path=image_path)
    annotations = read_single_image_annotations_csv_as_rc(path=annotations_path)
    return (image, annotations)

#------------------------------------------------------------------------------

def process_image_msct(
        path: str, subsample: int, 
        enrich_area_factor: float, enrich_max_mser: float,
        mser_percent_height: float, 
        segmentation_max_mser: float,
        segmentation_max_area_factor: float,
        segmentation_min_area: int
    ):
    '''
    Builds a multi-scale component-tree (MSCT) on an image using a given number of subsampling and enrichment scales.
    Extracts single objects from enriched nodes.

            Parameters:
                    `path` (str): path to the input directory containing a TIF image and a CSV dot annotations
                    `subsample` (int): number of subsampling steps (scales) to use
                    `enrich_area_factor` (float): percentage of the total image area allowed fror enrighment process [0,1]
                    `enrich_max_mser` (float): maximum MSER value for node selection
                    `mser_percent_height` (float): percentage of the MSCT height to use for MSER computation [0,1]
                    `segmentation_max_mser` (float): max MSER value when attempting to segment cells 
                    `segmentation_max_area_factor` (float): max area for chosen nodes
                    `segmentation_min_area` (int): minimum surface area of a flatzone to be considered a nucleus
    '''
    # get the data (image, annotations)
    #channel_images, annotations = get_image_and_annotations_from_dir(input_dir=path)
    channel_images = load_multi_channel_image(path=path)
    max_area = int(channel_images[0].shape[0] * channel_images[0].shape[1] * enrich_area_factor)
    output_dir = 'output'
    create_dir_remove_if_exist(output_dir)

    reconstructed_dir = os.path.join(output_dir, 'reconstructed')
    subsampled_dir = os.path.join(output_dir, 'subsampled')
    msct_dir = os.path.join(output_dir, 'msct')
    fz_dir = os.path.join(output_dir, 'flatzones')
    channel_dir = os.path.join(output_dir, 'channels')
    objects_dir = os.path.join(output_dir, 'objects')
    create_dir_remove_if_exist(reconstructed_dir)
    create_dir_remove_if_exist(subsampled_dir)
    create_dir_remove_if_exist(msct_dir)
    create_dir_remove_if_exist(fz_dir)
    create_dir_remove_if_exist(channel_dir)
    create_dir_remove_if_exist(objects_dir)

    wrapper = MSCTMultiChannelWrapper()
    #wrapper.set_annotations(annotations)
    msct_times = []
    t = time.time()

    # Step 1 : subsampling steps and SSCT building on low scale image
    wrapper.downsample_image(
        channel_images=channel_images, base_channel=0, 
        n=subsample, method='maximum', 
        use_median_filter=False, median_filter_size=3,
        invert_image=False, normalize=False
    )

    wrapper.save_subsampled_images(subsampled_dir=subsampled_dir, subsampled_name='subsampled')
    wrapper.save_channel_images(channel_dir)
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
    wrapper.reconstruct_image(reconstructed_dir=reconstructed_dir, reconstructed_name='rec_0', normalize=False)

    # STEP 1 : enrichment steps
    for subsample_step in range(0, subsample):

        # augmenting MSCT by one scale
        t = time.time()
        wrapper.augment_msct_mser(max_area=max_area, max_mser=enrich_max_mser)
        msct_times.append((f"enrich_{subsample_step}", time.time() - t))
        t = time.time()

        # reconstructing the original image with the MSCT
        wrapper.reconstruct_image(reconstructed_dir=reconstructed_dir, reconstructed_name=f"rec_{subsample_step+1}", normalize=False)

        # computing MSER
        t = time.time()
        wrapper.compute_mser_percent_height(percent_height=mser_percent_height)
        msct_times.append((f"mser_{subsample_step}", time.time() - t))

    # adding histogram
    t = time.time()
    wrapper.populate_histograms_with_channel_images()
    msct_times.append(("histogram", time.time() - t))
    t = time.time()

    for txt, val in msct_times:
        print(f"{txt} : {val}")
    total_time = sum([i[1] for i in msct_times])
    print(f"MSCT building time : {total_time}")

    # STEP 2 : segmentation of custers into objects (cells/nuclei)
    t = time.time()
    object_images, channel_vectors = wrapper.divide_objects(
        max_mser=segmentation_max_mser, 
        max_area_factor=segmentation_max_area_factor,
        min_area=segmentation_min_area
    )
    seg_time = time.time() - t
    print(f"MSCT segmentation time : {seg_time}")
    wrapper.export_objects_as_images(images=object_images, output_dir=objects_dir, all=True)
    wrapper.export_objects_as_label_image(images=object_images, output_dir=output_dir)
    wrapper.export_channel_vectors_as_array(vectors=channel_vectors, output_dir=output_dir)

#------------------------------------------------------------------------------

if __name__ == '__main__':

    # execute only if run as a script
    ap = argparse.ArgumentParser(prog="Multi-scale component-tree")
    ap.add_argument('indir', help='Input directory with multi channel TIF image and CSV dot annotations')
    ap.add_argument('subsample', help='Subsample steps')
    ap.add_argument('--mser_percent_height', help='Percentage of MSCT height used in MSER computation [0,1]', default=0.1)
    ap.add_argument('--enrich_area_factor', help='Maximum flat zone area to be enriched in percent of image size [0,1]', default=0.002)
    ap.add_argument('--enrich_max_mser', help='Maximum MSER value for candidate selection', default=1.0)
    ap.add_argument('--segmentation_max_mser', help='Maximum MSER value for the segentation step', default=1.0)
    ap.add_argument('--segmentation_max_area_factor', help='Maximum surface area for the segentation step', default=0.002)
    ap.add_argument('--segmentation_min_area', help='Minimum surface area for a nucleus', default=100)

    args = vars(ap.parse_args())
    process_image_msct(
        path=args['indir'], 
        subsample=int(args['subsample']), 
        enrich_area_factor=float(args['enrich_area_factor']),
        enrich_max_mser=float(args['enrich_max_mser']),
        mser_percent_height=float(args['mser_percent_height']),
        segmentation_max_mser=float(args['segmentation_max_mser']), 
        segmentation_max_area_factor=float(args['segmentation_max_area_factor']),
        segmentation_min_area=int(args['segmentation_min_area'])
    )