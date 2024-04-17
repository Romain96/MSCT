#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import argparse
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from collections import OrderedDict

from msct_wrapper import MSCTWrapper
from multi_scale_component_tree import MultiScaleComponentTree
from multi_scale_component_tree_node import MultiScaleComponentTreeNode
from timeit import default_timer as timer
from synthgen import Synthgen

#------------------------------------------------------------------------------

def get_nunber_of_stored_pixels(tree: MultiScaleComponentTree) -> int:
    root = tree.get_root()
    to_process = [root]
    nb = 0
    while len(to_process) > 0:
        node = to_process.pop(0)
        nb += len(node.get_pixels())
        for child in node.get_children():
            to_process.append(child)
    return nb

#------------------------------------------------------------------------------

def eval_temporal_and_spatial_maxtree(image: np.ndarray):
    '''
    Temporal measurements of building a standard max-tree on an input image.

            Parameters:
                    `image` (ndarray): input grayscale image
            
            Returns:
                    `mt_time` (float): elapsed time
                    `mt_nb_nodes` (int): number of built nodes
                    `mt_nb_pixels` (int): number of stored pixels
    '''
    # temporal
    t = timer()
    wrapper_mt = MSCTWrapper()
    wrapper_mt.subsample_image(image=image, n=0, method='maximum', use_median_filter=False, median_filter_size=3, invert_image=False)
    wrapper_mt.build_base_msct()
    mt_time = timer() - t
    # spatial
    tree = wrapper_mt.get_msct()
    mt_nb_nodes = len(tree.get_nodes())
    mt_nb_pixels = get_nunber_of_stored_pixels(tree)

    return (mt_time, mt_nb_nodes, mt_nb_pixels)
    

#------------------------------------------------------------------------------

def eval_temporal_and_spatial_msct(image: np.ndarray, k: int):
    '''
    Temporal and spatial measurements of building a MSCT on a set of downsampled image from a given input image.

            Parameters:
                    `image` (ndarray): input grayscale image
                    `k` (int): number of downsampling steps / enrichment steps
            
            Returns:
                    `msct_time` (float): elapsed time for MSCT building
                    `msct_nb_nodes` (int): number of built nodes
                    `msct_nb_pixels` (int): number of stored pixels
    '''
    # temporal
    t = timer() 
    #max_area = int(image.shape[0] * image.shape[1] * 0.10)
    max_area = int(image.shape[0] * image.shape[1])
    wrapper_msct = MSCTWrapper()
    wrapper_msct.subsample_image(image=image, n=k, method='maximum', use_median_filter=True, median_filter_size=3, invert_image=False)
    wrapper_msct.build_base_msct()
    wrapper_msct.compute_mser_percent_height(percent_height=0.1)
    for subsample_step in range(0, k):
        wrapper_msct.augment_msct_mser(max_area=max_area, max_mser=1.5)
        wrapper_msct.compute_mser_percent_height(percent_height=0.1)
    msct_time = timer() - t
    # spatial
    tree = wrapper_msct.get_msct()
    msct_nb_nodes = len(tree.get_nodes())
    msct_nb_pixels = get_nunber_of_stored_pixels(tree)

    return (msct_time, msct_nb_nodes, msct_nb_pixels)

#------------------------------------------------------------------------------

def eval_temporal_and_spatial_msct_and_seg(image: np.ndarray, k: int):
    '''
    Temporal and spatial measurements of building a MSCT on a set of downsampled
    image from a given input image and the segmentation step.

            Parameters:
                    `image` (ndarray): input grayscale image
                    `k` (int): number of downsampling steps / enrichment steps
            
            Returns:
                    `msct_time` (float): elapsed time for MSCT building
                    `msct_time_seg` (float): elapsed time for MSCT segmentation
                    `msct_nb_nodes` (int): number of built nodes
                    `msct_nb_pixels` (int): number of stored pixels
    '''
    # temporal
    t = timer() 
    max_area = int(image.shape[0] * image.shape[1] * 0.10)
    wrapper_msct = MSCTWrapper()
    wrapper_msct.subsample_image(image=image, n=k, method='maximum', use_median_filter=True, median_filter_size=3, invert_image=False)
    wrapper_msct.build_base_msct()
    wrapper_msct.compute_mser_percent_height(percent_height=0.1)
    for subsample_step in range(0, k):
        wrapper_msct.augment_msct_mser(max_area=max_area, max_mser=1.5)
        wrapper_msct.compute_mser_percent_height(percent_height=0.1)
    msct_time = timer() - t  
    t = timer()
    objects = wrapper_msct.divide_objects(max_mser=1.5, min_area=16)
    msct_time_seg = timer() - t

    # spatial
    tree = wrapper_msct.get_msct()
    msct_nb_nodes = len(tree.get_nodes())
    msct_nb_pixels = get_nunber_of_stored_pixels(tree)

    return (msct_time, msct_time_seg, msct_nb_nodes, msct_nb_pixels)

#------------------------------------------------------------------------------

def eval_average_temporal_and_spatial_maxtree(image: np.ndarray, n: int):
    '''
    Temporal and spatial measurements of building a standard max-tree on an input image.
    Results are averaged over `n` repetitions.

            Parameters:
                    `image` (ndarray): input grayscale image
                    `n` (int): number of repetitions
            
            Returns:
                    `avr_mt_time` (float): average elapsed time
                    `avr_mt_nb_nodes` (int): average number of nodes
                    `avr_mt_nb_pixels` (int): average number of stored pixels
    '''
    avr_mt_time = 0
    avr_mt_nb_nodes = 0
    avr_mt_nb_pixels = 0

    for i in range(0, n):
        mt_time, mt_nb_nodes, mt_nb_pixels = eval_temporal_and_spatial_maxtree(image)
        avr_mt_time += mt_time
        avr_mt_nb_nodes += mt_nb_nodes
        avr_mt_nb_pixels += mt_nb_pixels
    avr_mt_time = avr_mt_time / n
    avr_mt_nb_nodes = avr_mt_nb_nodes // n
    avr_mt_nb_pixels = avr_mt_nb_pixels // n

    return (avr_mt_time, avr_mt_nb_nodes, avr_mt_nb_pixels)

#------------------------------------------------------------------------------

def eval_average_temporal_and_spatial_msct(image: np.ndarray, k: int, n: int):
    '''
    Temporal and spatial measurements of building a MSCT on a set of downsampled
    image from a given input image. Results are averaged over `n` repetitions.

            Parameters:
                    `image` (ndarray): input grayscale image
                    `k` (int): number of downsampling steps / enrichment steps
                    `n` (int): number of repetitions
            
            Returns:
                    `avr_msct_time` (float): average elapsed time
                    `avr_msct_nb_nodes` (int): average number of nodes
                    `avr_msct_nb_pixels` (int): average number of stored pixels
    '''
    avr_msct_time = 0
    avr_msct_nb_nodes = 0
    avr_msct_nb_pixels = 0

    for i in range(0, n):
        msct_time, msct_nb_nodes, msct_nb_pixels = eval_temporal_and_spatial_msct(image=image, k=k)
        avr_msct_time += msct_time
        avr_msct_nb_nodes += msct_nb_nodes
        avr_msct_nb_pixels += msct_nb_pixels
    avr_msct_time = avr_msct_time / n
    avr_msct_nb_nodes = avr_msct_nb_nodes // n
    avr_msct_nb_pixels = avr_msct_nb_pixels // n

    return (avr_msct_time, avr_msct_nb_nodes, avr_msct_nb_pixels)

#------------------------------------------------------------------------------

def eval_average_temporal_and_spatial_msct_and_seg(image: np.ndarray, k: int, n: int):
    '''
    Temporal and spatial measurements of building a MSCT on a set of downsampled
    image from a given input image. Results are averaged over `n` repetitions.

            Parameters:
                    `image` (ndarray): input grayscale image
                    `k` (int): number of downsampling/enrichment steps
                    `n` (int): number of repetitions
            
            Returns:
                    `avr_msct_time` (float): average elapsed time for building the MSCT
                    `avr_msct_seg_time` (float) average elapsed time for segmentation
                    `avr_msct_nb_nodes` (int): average number of nodes
                    `avr_msct_nb_pixels` (int): average number of stored pixels
    '''
    avr_msct_time = 0
    avr_msct_seg_time = 0
    avr_msct_nb_nodes = 0
    avr_msct_nb_pixels = 0

    for i in range(0, n):
        msct_time, msct_seg_time, msct_nb_nodes, msct_nb_pixels = eval_temporal_and_spatial_msct_and_seg(image=image, k=k)
        avr_msct_time += msct_time
        avr_msct_seg_time += msct_seg_time
        avr_msct_nb_nodes += msct_nb_nodes
        avr_msct_nb_pixels += msct_nb_pixels
    avr_msct_time = avr_msct_time / n
    avr_msct_seg_time = avr_msct_seg_time / n
    avr_msct_nb_nodes = avr_msct_nb_nodes // n
    avr_msct_nb_pixels = avr_msct_nb_pixels // n

    return (avr_msct_time, avr_msct_seg_time, avr_msct_nb_nodes, avr_msct_nb_pixels)

#------------------------------------------------------------------------------

from PIL import Image
def save_image(image: np.ndarray, filename: str):
    img = Image.fromarray(image)
    img.save(filename)

def eval_temporal_and_spatial_over_multiple_images(
    k: int, nb_images: int, nb_repetitions: int, 
    image_size: int, nb_objects: int, object_size: int, object_size_spread: float, noise_variance: float
):
    mt_avr_time = 0
    msct_avr_time = 0
    mt_avr_nb_nodes = 0
    msct_avr_nb_nodes = 0
    mt_avr_nb_pixels = 0
    msct_avr_nb_pixels = 0

    for image_index in range(0, nb_images):
        # generate synthetic image
        image, _ = Synthgen.generate_synthetic_image(image_size, nb_objects, object_size, object_size_spread, noise_variance, False)
        save_image(image, f"size_{image_size}_iter_{image_index}.png")
        # evaluate (temporal & spatial)
        mt_time, mt_nb_nodes, mt_nb_pixels = eval_average_temporal_and_spatial_maxtree(image=image, n=nb_repetitions)
        msct_time, msct_nb_nodes, msct_nb_pixels = eval_average_temporal_and_spatial_msct(image=image, k=k, n=nb_repetitions)
        # sum
        mt_avr_time += mt_time
        msct_avr_time += msct_time
        mt_avr_nb_nodes += mt_nb_nodes
        msct_avr_nb_nodes += msct_nb_nodes
        mt_avr_nb_pixels += mt_nb_pixels
        msct_avr_nb_pixels += msct_nb_pixels

    mt_avr_time = mt_avr_time / nb_images
    msct_avr_time = msct_avr_time / nb_images
    mt_avr_nb_nodes = int(round(mt_avr_nb_nodes / nb_images))
    msct_avr_nb_nodes = int(round(msct_avr_nb_nodes / nb_images))
    mt_avr_nb_pixels = int(round(mt_avr_nb_pixels / nb_images))
    msct_avr_nb_pixels = int(round(msct_avr_nb_pixels / nb_images))

    return (mt_avr_time, msct_avr_time, mt_avr_nb_nodes, msct_avr_nb_nodes, mt_avr_nb_pixels, msct_avr_nb_pixels)

#------------------------------------------------------------------------------

def test_variable_image_size(
        k: int, 
        nb_images: int, 
        nb_repetitions: int, 
        nb_objects: int, 
        object_size: int, 
        object_size_spread: float, 
        noise_variance: float
    ):
    '''
    Test performances with image sizes ranging from 10 000 pixels to 500 000 pixels.
    The number and sizes of objects are fixed as well as the amount of noise.
    '''
    # building time
    mt_time_per_size = []
    msct_time_per_size = []
    # number of nodes (overall)
    mt_nb_nodes_per_size = []
    msct_nb_nodes_per_size = []
    # number of stored pixels (all scales)
    mt_nb_pixels_per_size = []
    msct_nb_pixels_per_size = []

    sizes = [10000 * i for i in range(1, 41)]
    for size in sizes:
        print(size)

        mt_time, msct_time, mt_nb_nodes, msct_nb_nodes, mt_nb_pixels, msct_nb_pixels = eval_temporal_and_spatial_over_multiple_images(
            k=k, 
            nb_images=nb_images, 
            nb_repetitions=nb_repetitions, 
            image_size=size, 
            nb_objects=nb_objects, 
            object_size=object_size, 
            object_size_spread=object_size_spread, 
            noise_variance=noise_variance
        )

        mt_time_per_size.append(mt_time)
        msct_time_per_size.append(msct_time)
        mt_nb_nodes_per_size.append(mt_nb_nodes)
        msct_nb_nodes_per_size.append(msct_nb_nodes)
        mt_nb_pixels_per_size.append(mt_nb_pixels)
        msct_nb_pixels_per_size.append(msct_nb_pixels)

    plt.clf()
    plt.cla()
    plt.title("MSCT vs max-tree")
    plt.xlabel("Image size (pixels)")
    plt.ylabel("Elapsed time (s)")
    plt.scatter(sizes, mt_time_per_size, marker='.', color='red', label='max-tree')
    plt.scatter(sizes, msct_time_per_size, marker='.', color='blue', label='MSCT')
    plt.legend()
    plt.savefig("measure_temporal.png")

    plt.clf()
    plt.cla()
    plt.title("MSCT vs max-tree")
    plt.xlabel("Image size (pixels)")
    plt.ylabel("Number of created nodes")
    plt.scatter(sizes, mt_nb_nodes_per_size, marker='.', color='red', label='max-tree')
    plt.scatter(sizes, msct_nb_nodes_per_size, marker='.', color='blue', label='MSCT')
    plt.legend()
    plt.savefig("measure_spatial_nodes.png")

    plt.clf()
    plt.cla()
    plt.title("MSCT vs max-tree")
    plt.xlabel("Image size (pixels)")
    plt.ylabel("Number of stored pixels")
    plt.scatter(sizes, mt_nb_pixels_per_size, marker='.', color='red', label='max-tree')
    plt.scatter(sizes, msct_nb_pixels_per_size, marker='.', color='blue', label='MSCT')
    plt.legend()
    plt.savefig("measure_spatial_pixels.png")

#------------------------------------------------------------------------------

def test_nb_cells(obj_min: int, obj_max: int, radius: int, radius_spread: float, noise_var: float):
    '''
    Tests the MSCT/max-tree performances on a single size but with increasing number of cells
    '''
    size = 90000 # aka 300 x 300 images
    coverings = OrderedDict()
    cell_sizes = [i for i in range(obj_min, obj_max + 1, 1)]

    for nb_cells in cell_sizes:
        image = Synthgen.generate_synthetic_image_random(size, nb_cells, radius, radius_spread, noise_var)
        avr_mt_time = 0; avr_msct_time = 0
        avr_nb_nodes_mt = 0; avr_nb_nodes_msct = 0
        avr_nb_pixels_mt = 0; avr_nb_pixels_msct = 0
        repetitions = 1
        for rep in range(0, repetitions):
            print(f"nb_cells = {nb_cells} ({obj_min}-{obj_max}), repetition {rep+1}/{repetitions}")
            mt_time, mt_nb_nodes, mt_nb_pixels = eval_temporal_and_spatial_maxtree(image)
            msct_time, msct_nb_nodes, msct_nb_pixels = eval_temporal_and_spatial_msct(image, 3)
            avr_mt_time += mt_time; avr_msct_time += msct_time
            avr_nb_nodes_mt += mt_nb_nodes; avr_nb_nodes_msct += msct_nb_nodes
            avr_nb_pixels_mt += mt_nb_pixels; avr_nb_pixels_msct += msct_nb_pixels
        avr_mt_time = avr_mt_time / repetitions; avr_msct_time = avr_msct_time / repetitions
        avr_nb_nodes_mt = avr_nb_nodes_mt / repetitions; avr_nb_nodes_msct = avr_nb_nodes_msct / repetitions
        avr_nb_pixels_mt = avr_nb_pixels_mt / repetitions; avr_nb_pixels_msct = avr_nb_pixels_msct / repetitions
        mask = np.zeros_like(image, dtype=bool)
        mask[np.where(image >= 75)] = True
        covering = np.count_nonzero(mask) / size
        coverings[covering*100] = {
            'time_mt': avr_mt_time,
            'time_msct': avr_msct_time,
            'nodes_mt': avr_nb_nodes_mt,
            'nodes_msct': avr_nb_nodes_msct,
            'pixels_mt': avr_nb_pixels_mt,
            'pixels_msct': avr_nb_pixels_msct
        }

    covers = [key for key in coverings.keys()]
    mt_times = [coverings[key]['time_mt'] for key in coverings.keys()]
    msct_times = [coverings[key]['time_msct'] for key in coverings.keys()]
    '''
    mt_fit_time = np.polyfit(covers, mt_times, 2)
    msct_fit_time = np.polyfit(covers, msct_times, 2)
    f_mt_time = np.poly1d(mt_fit_time)
    f_msct_time = np.poly1d(msct_fit_time)'''

    mt_nodes = [coverings[key]['nodes_mt'] for key in coverings.keys()]
    msct_nodes = [coverings[key]['nodes_msct'] for key in coverings.keys()]
    '''
    mt_fit_nodes = np.polyfit(covers, mt_nodes, 2)
    msct_fit_nodes = np.polyfit(covers, msct_nodes, 2)
    f_mt_nodes = np.poly1d(mt_fit_nodes)
    f_msct_nodes = np.poly1d(msct_fit_nodes)
    '''

    mt_pixels = [coverings[key]['pixels_mt'] for key in coverings.keys()]
    msct_pixels = [coverings[key]['pixels_msct'] for key in coverings.keys()]
    '''
    mt_fit_pixels = np.polyfit(covers, mt_pixels, 2)
    msct_fit_pixels = np.polyfit(covers, msct_pixels, 2)
    f_mt_pixels = np.poly1d(mt_fit_pixels)
    f_msct_pixels = np.poly1d(msct_fit_pixels)

    fit_x = np.linspace(covers[0], covers[-1], 200)
    fit_y_mt_time = f_mt_time(fit_x)
    fit_y_msct_time = f_msct_time(fit_x)
    fit_y_mt_nodes = f_mt_nodes(fit_x)
    fit_y_msct_nodes = f_msct_nodes(fit_x)
    fit_y_mt_pixels = f_mt_pixels(fit_x)
    fit_y_msct_pixels = f_msct_pixels(fit_x)'''

    # time relative to minimum time + differential
    mt_min_time = min(mt_times)
    mt_times_relative = (np.array(mt_times) - mt_min_time) / mt_min_time * 100
    mt_times_base_0 = [0] * len(mt_times)
    #msct_times_relative = (np.array(msct_times) - mt_min_time) / mt_min_time * 100
    msct_times_relative = (np.array(msct_times) - np.array(mt_times)) / np.array(mt_times) * 100

    plt.clf()
    plt.cla()
    plt.scatter(covers, mt_times_base_0, marker='.', color='red', label='max-tree')
    plt.scatter(covers, msct_times_relative, marker='.', color='blue', label='MSCT')
    plt.title("MSCT vs max-tree")
    plt.xlabel("Foreground to background ratio (%)")
    plt.ylabel("Elapsed time difference (%)")
    plt.legend()
    plt.savefig("measure_temporal.svg")

    plt.clf()
    plt.cla()
    plt.scatter(covers, mt_nodes, marker='.', color='red', label='max-tree')
    plt.scatter(covers, msct_nodes, marker='.', color='blue', label='MSCT')
    plt.title("MSCT vs max-tree")
    plt.xlabel("Foreground to background ratio (%)")
    plt.ylabel("Number of created nodes")
    plt.legend()
    plt.savefig("measure_spatial_nodes.svg")

    plt.clf()
    plt.cla()
    plt.scatter(covers, mt_pixels, marker='.', color='red', label='max-tree')
    plt.scatter(covers, msct_pixels, marker='.', color='blue', label='MSCT')
    plt.title("MSCT vs max-tree")
    plt.xlabel("Foreground to background ratio (%)")
    plt.ylabel("Number of stored pixels")  
    plt.legend()
    plt.savefig("measure_spatial_pixels.svg")

    '''
    # saving temporal measures - fit & raw
    plt.clf()
    plt.cla()
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    axes[0].plot(fit_x, fit_y_mt_time, color='red', label='max-tree')
    axes[0].plot(fit_x, fit_y_msct_time, color='blue', label='MSCT')
    axes[1].scatter(covers, mt_times, marker='.', color='red', label='max-tree')
    axes[1].scatter(covers, msct_times, marker='.', color='blue', label='MSCT')
    plt.title("MSCT vs max-tree")
    plt.xlabel("Foreground to background ratio (%)")
    plt.ylabel("Elapsed time (s)")
    plt.legend()
    plt.savefig("measure_temporal.png")

    # saving spatial (nodes) measures - fit
    plt.clf()
    plt.cla()
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    axes[0].plot(fit_x, fit_y_mt_nodes, color='red', label='max-tree')
    axes[0].plot(fit_x, fit_y_msct_nodes, color='blue', label='MSCT')
    axes[1].scatter(covers, mt_nodes, marker='.', color='red', label='max-tree')
    axes[1].scatter(covers, msct_nodes, marker='.', color='blue', label='MSCT')
    plt.title("MSCT vs max-tree")
    plt.xlabel("Foreground to background ratio (%)")
    plt.ylabel("Number of created nodes")
    plt.legend()
    plt.savefig("measure_spatial_nodes.png")

    # saving spatial (pixels) measures - fit
    plt.clf()
    plt.cla()
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    axes[0].plot(fit_x, fit_y_mt_pixels, color='red', label='max-tree')
    axes[0].plot(fit_x, fit_y_msct_pixels, color='blue', label='MSCT')
    axes[1].scatter(covers, mt_pixels, marker='.', color='red', label='max-tree')
    axes[1].scatter(covers, msct_pixels, marker='.', color='blue', label='MSCT')
    plt.title("MSCT vs max-tree")
    plt.xlabel("Foreground to background ratio (%)")
    plt.ylabel("Number of stored pixels")  
    plt.legend()
    plt.savefig("measure_spatial_pixels.png")'''

#------------------------------------------------------------------------------

def test_nb_cells_and_cell_size():
    '''
    Tests the max-tree performances on a single size but with increasing number of cells and cell sizes
    '''
    image_size = 90000 # aka 300 x 300 images
    all_nb_cells = [i for i in range(5, 105, 5)]
    all_cell_size = [i for i in range(3, 11)]
    curves_time_mt = []; raw_mt_times = []
    curves_time_msct = []; raw_msct_times = []
    curves_nodes_mt = []; raw_mt_nodes = []
    curves_nodes_msct = []; raw_msct_nodes = []
    curves_pixels_mt = []; raw_mt_pixels = []
    curves_pixels_msct = []; raw_msct_pixels = []
    
    # generating one curve for each cell size
    for cell_size in all_cell_size:

        print(f"cell_size : {cell_size}")
        y_values_mt = OrderedDict()
        y_values_msct = OrderedDict()

        # generating points for the curve with fixed cell size
        for nb_cells in all_nb_cells:

            print(f"\tnb_cells : {nb_cells}")
            # generating a synthetic image of fixed size for the given number of cells and cell size (spread is null)
            image, _ = Synthgen.generate_synthetic_image(image_size, nb_cells, cell_size, 0, 0.001, False)

            avr_mt_time = 0; avr_msct_time = 0
            avr_nb_nodes_mt = 0; avr_nb_nodes_msct = 0
            avr_nb_pixels_mt = 0; avr_nb_pixels_msct = 0
            repetitions = 5
        
            for rep in range(0, repetitions):

                # max-tree
                mt_time, mt_nb_nodes, mt_nb_pixels = eval_temporal_and_spatial_maxtree(image)
                avr_mt_time += mt_time
                avr_nb_nodes_mt += mt_nb_nodes
                avr_nb_pixels_mt += mt_nb_pixels

                # MSCT
                msct_time, msct_nb_nodes, msct_nb_pixels = eval_temporal_and_spatial_msct(image, 2)
                avr_msct_time += msct_time
                avr_nb_nodes_msct += msct_nb_nodes
                avr_nb_pixels_msct += msct_nb_pixels

            # max-tree
            avr_mt_time = avr_mt_time / repetitions
            avr_nb_nodes_mt = avr_nb_nodes_mt / repetitions
            avr_nb_pixels_mt = avr_nb_pixels_mt / repetitions
            y_values_mt[nb_cells] = {
                'time_mt': avr_mt_time,
                'nodes_mt': avr_nb_nodes_mt,
                'pixels_mt': avr_nb_pixels_mt
            }

            # MSCT
            avr_msct_time = avr_msct_time / repetitions
            avr_nb_nodes_msct = avr_nb_nodes_msct / repetitions
            avr_nb_pixels_msct = avr_nb_pixels_msct / repetitions
            y_values_msct[nb_cells] = {
                'time_msct': avr_msct_time,
                'nodes_msct': avr_nb_nodes_msct,
                'pixels_msct': avr_nb_pixels_msct
            }
    
        # max-tree
        covers = [key for key in y_values_mt.keys()]
        mt_times = [y_values_mt[key]['time_mt'] for key in y_values_mt.keys()]
        mt_nodes = [y_values_mt[key]['nodes_mt'] for key in y_values_mt.keys()]
        mt_pixels = [y_values_mt[key]['pixels_mt'] for key in y_values_mt.keys()]
        mt_fit_time = np.polyfit(covers, mt_times, 2)
        mt_fit_nodes = np.polyfit(covers, mt_nodes, 2)
        mt_fit_pixels = np.polyfit(covers, mt_pixels, 2)
        f_mt_time = np.poly1d(mt_fit_time)
        f_mt_nodes = np.poly1d(mt_fit_nodes)
        f_mt_pixels = np.poly1d(mt_fit_pixels)
        curves_time_mt.append(f_mt_time); raw_mt_times.append(mt_times)
        curves_nodes_mt.append(f_mt_nodes); raw_mt_nodes.append(mt_nodes)
        curves_pixels_mt.append(f_mt_pixels); raw_mt_pixels.append(mt_pixels)

        # MSCT
        msct_times = [y_values_msct[key]['time_msct'] for key in y_values_msct.keys()]
        msct_nodes = [y_values_msct[key]['nodes_msct'] for key in y_values_msct.keys()]
        msct_pixels = [y_values_msct[key]['pixels_msct'] for key in y_values_msct.keys()]
        msct_fit_time = np.polyfit(covers, msct_times, 2)
        msct_fit_nodes = np.polyfit(covers, msct_nodes, 2)
        msct_fit_pixels = np.polyfit(covers, msct_pixels, 2)
        f_msct_time = np.poly1d(msct_fit_time)
        f_msct_nodes = np.poly1d(msct_fit_nodes)
        f_msct_pixels = np.poly1d(msct_fit_pixels)
        curves_time_msct.append(f_msct_time); raw_msct_times.append(msct_times)
        curves_nodes_msct.append(f_msct_nodes); raw_msct_nodes.append(msct_nodes)
        curves_pixels_msct.append(f_msct_pixels); raw_msct_pixels.append(msct_pixels)

    # temporal measures - fit & raw
    plt.clf()
    plt.cla()
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    fit_x = np.linspace(covers[0], covers[-1], 200)
    color = iter(cm.rainbow(np.linspace(0, 1, len(curves_time_mt))))
    for index in range(0, len(all_cell_size)):
        local_cell_size = all_cell_size[index]
        local_colour = next(color)  
        # max-tree
        fit_y_mt = curves_time_mt[index](fit_x)
        raw_y_mt = raw_mt_times[index]
        axes[0].scatter(covers, raw_y_mt, marker='.', color=local_colour)#, label=f"MT r={local_cell_size}")
        axes[1].plot(fit_x, fit_y_mt, linestyle='solid', color=local_colour, label=f"MT r={local_cell_size}")
        # msct
        fit_y_msct = curves_time_msct[index](fit_x)
        raw_y_msct = raw_msct_times[index]
        axes[0].scatter(covers, raw_y_msct, marker='x', color=local_colour)#, label=f"MT r={local_cell_size}")
        axes[1].plot(fit_x, fit_y_msct, linestyle='dashed', color=local_colour, label=f"MSCT r={local_cell_size}")
        plt.plot()
    plt.title("max-tree vs MSCT performances")
    plt.xlabel("Number of cells")
    plt.ylabel("Elapsed time (s)")
    fig.legend(loc='outside right upper')
    plt.savefig("measure_temporal.png")

    # temporal spatial (nodes) - fit & raw
    plt.clf()
    plt.cla()
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    fit_x = np.linspace(covers[0], covers[-1], 200)
    color = iter(cm.rainbow(np.linspace(0, 1, len(curves_nodes_mt))))
    for index in range(0, len(all_cell_size)):
        local_cell_size = all_cell_size[index]
        local_colour = next(color)  
        # max-tree
        fit_y_mt = curves_nodes_mt[index](fit_x)
        raw_y_mt = raw_mt_nodes[index]
        axes[0].scatter(covers, raw_y_mt, marker='.', color=local_colour)#, label=f"MT r={local_cell_size}")
        axes[1].plot(fit_x, fit_y_mt, linestyle='solid', color=local_colour, label=f"MT r={local_cell_size}")
        # msct
        fit_y_msct = curves_nodes_msct[index](fit_x)
        raw_y_msct = raw_msct_nodes[index]
        axes[0].scatter(covers, raw_y_msct, marker='x', color=local_colour)#, label=f"MT r={local_cell_size}")
        axes[1].plot(fit_x, fit_y_msct, linestyle='dashed', color=local_colour, label=f"MSCT r={local_cell_size}")
        plt.plot()
    plt.title("max-tree vs MSCT performances")
    plt.xlabel("Number of cells")
    plt.ylabel("Number of nodes")
    fig.legend(loc='outside right upper')
    plt.savefig("measure_spatial_nodes.png")

    # spatial measures (pixels) - fit & raw
    plt.clf()
    plt.cla()
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    fit_x = np.linspace(covers[0], covers[-1], 200)
    color = iter(cm.rainbow(np.linspace(0, 1, len(curves_time_mt))))
    for index in range(0, len(all_cell_size)):
        local_cell_size = all_cell_size[index]
        local_colour = next(color)  
        # max-tree
        fit_y_mt = curves_pixels_mt[index](fit_x)
        raw_y_mt = raw_mt_pixels[index]
        axes[0].scatter(covers, raw_y_mt, marker='.', color=local_colour)#, label=f"MT r={local_cell_size}")
        axes[1].plot(fit_x, fit_y_mt, linestyle='solid', color=local_colour, label=f"MT r={local_cell_size}")
        # msct
        fit_y_msct = curves_pixels_msct[index](fit_x)
        raw_y_msct = raw_msct_pixels[index]
        axes[0].scatter(covers, raw_y_msct, marker='x', color=local_colour)#, label=f"MT r={local_cell_size}")
        axes[1].plot(fit_x, fit_y_msct, linestyle='dashed', color=local_colour, label=f"MSCT r={local_cell_size}")
        plt.plot()
    plt.title("max-tree vs MSCT performances")
    plt.xlabel("Number of cells")
    plt.ylabel("Number of pixels")
    fig.legend(loc='outside right upper')
    plt.savefig("measure_spatial_pixels.png")

#------------------------------------------------------------------------------

def test_multiple_subdivision_factors(min_obj: int, max_obj: int, radius: int, radius_spread: float, noise_var: float):
    '''
    Tests the MSCT/max-tree performances on a single size with multiple subdivision factors
    MSCT 0 ~ MT, MSCT 1, MSCT 2 (standard), MSCT 3
    '''
    size = 90000 # aka 300 x 300 images
    all_nb_cells = [i for i in range(min_obj, max_obj + 1, 5)]
    repetitions = 5
    nb_subdivisions = 4

    size_avr_mt_time = []; size_all_avr_msct_time = [[] for i in range(0, nb_subdivisions)]
    size_avr_nb_nodes = []; size_all_avr_msct_nb_nodes = [[] for i in range(0, nb_subdivisions)]
    size_avr_nb_pixels = []; size_all_avr_msct_nb_pixels = [[] for i in range(0, nb_subdivisions)]

    for nb_cells in all_nb_cells:

        print(f"\tnb_cells : {nb_cells}/{max_obj}")
        avr_mt_time = 0; all_avr_msct_time = nb_subdivisions * [0]
        avr_nb_nodes_mt = 0; all_avr_nb_nodes_msct = nb_subdivisions * [0]
        avr_nb_pixels_mt = 0; all_avr_nb_pixels_msct = nb_subdivisions * [0]

        for rep in range(0, repetitions):

            # creating image
            image = Synthgen.generate_synthetic_image_random(size, nb_cells, radius, radius_spread, noise_var)

            # max-tree
            mt_time, mt_nb_nodes, mt_nb_pixels = eval_temporal_and_spatial_maxtree(image)
            avr_mt_time += mt_time
            avr_nb_nodes_mt += mt_nb_nodes
            avr_nb_pixels_mt += mt_nb_pixels
            # MSCT 0, 1, 2, 3
            for subdivision_factor in range(0, nb_subdivisions):
                msct_time, msct_nb_nodes, msct_nb_pixels = eval_temporal_and_spatial_msct(image, subdivision_factor)
                all_avr_msct_time[subdivision_factor] += msct_time
                all_avr_nb_nodes_msct[subdivision_factor] += msct_nb_nodes
                all_avr_nb_pixels_msct[subdivision_factor] += msct_nb_pixels

        # averages of repetitions  
        avr_mt_time = avr_mt_time / repetitions
        avr_nb_nodes_mt = avr_nb_nodes_mt / repetitions
        avr_nb_pixels_mt = avr_nb_pixels_mt / repetitions

        # new data point
        size_avr_mt_time.append(avr_mt_time)
        size_avr_nb_nodes.append(avr_nb_nodes_mt)
        size_avr_nb_pixels.append(avr_nb_pixels_mt)
        for i in range(0, nb_subdivisions):
            size_all_avr_msct_time[i].append(all_avr_msct_time[i] / repetitions)
            size_all_avr_msct_nb_nodes[i].append(all_avr_nb_nodes_msct[i] / repetitions)
            size_all_avr_msct_nb_pixels[i].append(all_avr_nb_pixels_msct[i] / repetitions)

    fit_x = np.linspace(all_nb_cells[0], all_nb_cells[-1], 200)

    # raw & fit max-tree
    mt_fit_time = np.polyfit(all_nb_cells, size_avr_mt_time, 2)
    mt_fit_nodes = np.polyfit(all_nb_cells, size_avr_nb_nodes, 2)
    mt_fit_pixels = np.polyfit(all_nb_cells, size_avr_nb_pixels, 2)
    f_mt_time = np.poly1d(mt_fit_time)
    f_mt_nodes = np.poly1d(mt_fit_nodes)
    f_mt_pixels = np.poly1d(mt_fit_pixels)
    fit_mt_time = f_mt_time(fit_x)
    fit_mt_nodes = f_mt_nodes(fit_x)
    fit_mt_pixels = f_mt_pixels(fit_x)

    # raw & fit MSCT 0, 1, 2, 3
    all_fit_msct_time = []; all_fit_msct_nodes = []; all_fit_msct_pixels = []
    for i in range(0, nb_subdivisions):
        msct_fit_time = np.polyfit(all_nb_cells, size_all_avr_msct_time[i], 2)
        msct_fit_nodes = np.polyfit(all_nb_cells, size_all_avr_msct_nb_nodes[i], 2)
        msct_fit_pixels = np.polyfit(all_nb_cells, size_all_avr_msct_nb_pixels[i], 2)
        f_msct_time = np.poly1d(msct_fit_time)
        f_msct_nodes = np.poly1d(msct_fit_nodes)
        f_msct_pixels = np.poly1d(msct_fit_pixels)
        all_fit_msct_time.append(f_msct_time(fit_x))
        all_fit_msct_nodes.append(f_msct_nodes(fit_x))
        all_fit_msct_pixels.append(f_msct_pixels(fit_x))

    # saving temporal measures - fit & raw
    plt.clf()
    plt.cla()
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    color = iter(cm.rainbow(np.linspace(0, 1, nb_subdivisions + 1)))
    # max-tree
    c = next(color)
    axes[0].plot(fit_x, fit_mt_time, color=c, linestyle='dashed', label='max-tree')
    axes[1].scatter(all_nb_cells, size_avr_mt_time, marker='.', color=c, label='max-tree')
    # MSCTs
    for i in range(0, nb_subdivisions):
        c = next(color)
        axes[0].plot(fit_x, all_fit_msct_time[i], color=c, label=f"MSCT {i}")
        axes[1].scatter(all_nb_cells, size_all_avr_msct_time[i], marker='.', color=c, label=f"MSCT {i}")
    plt.title("MSCT vs max-tree")
    plt.xlabel("Foreground to background ratio (%)")
    plt.ylabel("Elapsed time (s)")
    fig.legend(loc='outside right upper')
    plt.savefig("measure_temporal.png")

    # saving spatial (nodes) measures - fit
    plt.clf()
    plt.cla()
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    color = iter(cm.rainbow(np.linspace(0, 1, nb_subdivisions + 1)))
    # max-tree
    c = next(color)
    axes[0].plot(fit_x, fit_mt_nodes, color=c, linestyle='dashed', label='max-tree')
    axes[1].scatter(all_nb_cells, size_avr_nb_nodes, marker='.', color=c, label='max-tree')
    # MSCTs
    for i in range(0, nb_subdivisions):
        c = next(color)
        axes[0].plot(fit_x, all_fit_msct_nodes[i], color=c, label=f"MSCT {i}")
        axes[1].scatter(all_nb_cells, size_all_avr_msct_nb_nodes[i], marker='.', color=c, label=f"MSCT {i}")
    plt.title("MSCT vs max-tree")
    plt.xlabel("Foreground to background ratio (%)")
    plt.ylabel("Number of nodes")
    fig.legend(loc='outside right upper')
    plt.savefig("measure_spatial_nodes.png")

    # saving spatial (pixels) measures - fit
    plt.clf()
    plt.cla()
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    color = iter(cm.rainbow(np.linspace(0, 1, nb_subdivisions + 1)))
    # max-tree
    c = next(color)
    axes[0].plot(fit_x, fit_mt_pixels, color=c, linestyle='dashed', label='max-tree')
    axes[1].scatter(all_nb_cells, size_avr_nb_pixels, marker='.', color=c, label='max-tree')
    # MSCTs
    for i in range(0, nb_subdivisions):
        c = next(color)
        axes[0].plot(fit_x, all_fit_msct_pixels[i], color=c, label=f"MSCT {i}")
        axes[1].scatter(all_nb_cells, size_all_avr_msct_nb_pixels[i], marker='.', color=c, label=f"MSCT {i}")
    plt.title("MSCT vs max-tree")
    plt.xlabel("Foreground to background ratio (%)")
    plt.ylabel("Number of pixels")  
    fig.legend(loc='outside right upper')
    plt.savefig("measure_spatial_pixels.png")

#------------------------------------------------------------------------------

def test():
    #test_variable_image_size(k=2, nb_images=5, nb_repetitions=2, nb_objects=10, object_size=5, object_size_spread=0.2, noise_variance=0.001)
    test_nb_cells(5, 500, 10, 0.0, 0.001)
    #test_nb_cells_and_cell_size()
    #test_multiple_subdivision_factors(10, 500, 5, 0.2, 0.001)

#------------------------------------------------------------------------------

if __name__ == '__main__':

    # execute only if run as a script
    ap = argparse.ArgumentParser(prog="Multi-scale component-tree")

    sys.setrecursionlimit(1000000000)
    args = vars(ap.parse_args())

    test()
