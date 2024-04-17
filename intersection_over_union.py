#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import argparse
import numpy as np
import os
import shutil
import glob
import csv

from distinctipy import distinctipy
from msct_utils import load_image, convert_colour_to_grayscale, save_image

#------------------------------------------------------------------------------

def create_dir_remove_if_exist(directory: str, debug=False) -> None:
    '''
        Creates a directory if it does not already exists.

                Parameters:
                        `directory` (str): path to the directory
                        `debug` (bool): whether to print debug informations

                Returns:
                        None
        '''
    if debug:
        print(f"DEBUG {create_dir_remove_if_exist.__name__} : creating directory {directory}")
    if os.path.isdir(directory):
        if debug:
            print(f"DEBUG {create_dir_remove_if_exist.__name__} : directory exists, removing")
        shutil.rmtree(directory)
    os.mkdir(directory)

#------------------------------------------------------------------------------

def build_node_content_image(image: np.ndarray, pixels: set) -> np.ndarray:
    '''
    Creates a binary image with `pixels` representing the foreground.

            Parameters:
                    `image` (ndarray): image of the same dimensions as the result image to create
                    `pixels` (set): set of pixels (line, column) to assign to the foreground

            Returns:
                    `zone` (ndarray): result binary image of the same dimension as `image`
    '''
    zone = np.zeros(image.shape, dtype=image.dtype)
    for x, y in pixels:
        zone[x, y] = 255
    return zone

#------------------------------------------------------------------------------

def get_images_in_dir(directory: str, extension='png'):
    '''
    Returns a set of paths for all images in the given directory

            Parameters:
                    `directory (str)`: directory to explore
                    `extension (str)`: file extension

            Returns:
                    `images` (list): list of paths
    '''
    images = glob.glob(os.path.join(directory, f"*.{extension}"))
    return images

#------------------------------------------------------------------------------

def load_gt_masks(images: list[np.ndarray]) -> list:
    '''
    Loads all ground truth masks

            Parameters:
                    `path` (str): path to the images

            Returns:
                    `image` (list): loaded images
    '''
    all_images = []
    for image in images:
        loaded = load_image(image)
        loaded = convert_colour_to_grayscale(loaded)
        all_images.append(loaded.astype(bool))
    return all_images

#------------------------------------------------------------------------------

def load_pred_masks(images: list[str]) -> list[np.ndarray]:
    '''
    Loads all predicted masks

            Parameters:
                    `images` (list): path to the images

            Returns:
                    `all_images` (list): loaded images
    '''
    all_images = []
    for image in images:
        loaded = load_image(image)
        loaded = convert_colour_to_grayscale(loaded)
        all_images.append(loaded.astype(bool))
    return all_images

#------------------------------------------------------------------------------

def compute_intersection_over_union(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    '''
    Computes the Intersection over Union (IoU) between a ground truth mask and a predicted mask

            Parameters:
                    gt_mask (ndarray): ground truth mask for a single object
                    pred_mask (ndarray): predicted mask for a single object
                    `threshold` (float): threshold value over which the prediction is deemed acceptable

            Returns:
                    `iou` (float): Intersection over Union raw value
    '''
    # compute the intersection
    inter_image = np.logical_and(gt_mask, pred_mask)
    inter = np.count_nonzero(inter_image)
    # compute the union
    union_image = np.logical_or(gt_mask, pred_mask)
    union = np.count_nonzero(union_image)
    # compute the Intersection over Union = (A inter B) / (A union B)
    iou = inter / union
    return iou

#------------------------------------------------------------------------------

def compute_image_precision(gt_masks, pred_masks, thresholds):
    '''
    Computes the precision of a given images given a set of thresholds 
    and two sets of predicted and ground truth images.

            parameters:
                    `gt_masks` (list): list of ground truth images
                    `pred_masks` (list): list of predicted images
                    `thresholds` (list): list of threshold values in [0,1]

            Returns:
                `precision` (float): average precision over all thresholds
                `precision_per_threshold` (list): list of individual precisions for each threshold in `thresholds`
    '''

    tp_fp_fn_per_threshold = []
    tp_fp_fn_tn_per_threshold = []

    for threshold in thresholds:

        gt_to_pred = []
        for i in range(0, len(gt_masks)):
            gt_to_pred.append([])

        for gt_index in range(0, len(gt_masks)):
            gt_mask = gt_masks[gt_index]
            for pred_index in range(0, len(pred_masks)):
                pred_mask = pred_masks[pred_index]
                # compute IoU
                iou = compute_intersection_over_union(gt_mask, pred_mask)
                # if IoU > threshold the the association yields True Positive (TP)
                if iou > threshold:
                    gt_to_pred[gt_index].append((pred_index, iou))

        associated_preds = set()
        true_positives = set()
        false_positives = set()
        false_negatives = set()

        image_size = gt_masks[0].shape[0] * gt_masks[0].shape[1]
        tp_pix = 0
        fp_pix = 0
        fn_pix = 0
        tn_pix = image_size

        # for each ground truth mask, take the highest associated prediction and assigns as TP
        for gt_index in range(0, len(gt_to_pred)):
            candidates = gt_to_pred[gt_index]

            # if no candidate then gt is assigned False Negative (ground truth object with no associated predicted object)
            if len(candidates) == 0:
                false_negatives.add(gt_index)
                fn_pix += np.count_nonzero(gt_masks[gt_index])

            # if at least one candidate then if one valid candidate is above the threshold -> True Positive
            else:
                sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                for candidate, iou in sorted_candidates:
                    if iou > threshold and gt_index not in true_positives:
                        true_positives.add(gt_index)
                        tp_pix += np.count_nonzero(gt_masks[gt_index])
                        associated_preds.add(candidate)
                        break

        # unassigned predicted are False Positives
        for pred_index in range(0, len(pred_masks)):
            if pred_index not in associated_preds:
                false_positives.add(pred_index)
                fp_pix += np.count_nonzero(gt_masks[gt_index])
        
        tn_pix = tn_pix - tp_pix - fp_pix - fn_pix
        tp_pix /= image_size
        fp_pix /= image_size
        fn_pix /= image_size
        tn_pix /= image_size
        tp_fp_fn_tn_per_threshold.append((tp_pix, fp_pix, fn_pix, tn_pix))
        tp_fp_fn_per_threshold.append((len(true_positives), len(false_positives), len(false_negatives)))

    # computing the precision averaged over all threshold values
    precision_per_threshold = []
    prec_sum = 0
    for tp, fp, fn in tp_fp_fn_per_threshold:
        precision_per_threshold.append(tp / (tp + fp + fn))
        prec_sum += (tp / (tp + fp + fn))
    precision = prec_sum / len(tp_fp_fn_per_threshold)
    return precision, precision_per_threshold, tp_fp_fn_tn_per_threshold

#------------------------------------------------------------------------------

def evaluate_image(gt_dir: str, pred_dir: str, image_name: str, thresholds: list, output_dir: str) -> dict:
    '''
    Computes the precision for an image in using ground truth in `gt_dir` and predicted objects in `pred_dir`

            Parameters:
                    `gt_dir` (str): directory containing ground truth object images
                    `pred_dir` (str): directory containing predicted object images
                    `image_name` (str): name of the image without extension
                    `thresholds` (list): list of individual thresholds used to compute the intersection over union
                    `output_dir` (str): directory for outputting gt and pred visualisation images

            Returns:
                    `data` (dict): directory containing average precision and individual precisions for the evaluated image
    '''
    gt_paths = get_images_in_dir(os.path.join(gt_dir, image_name, 'masks'))
    pred_paths = get_images_in_dir(os.path.join(pred_dir, image_name, 'objects'))
    vis_img = os.path.join(pred_dir, image_name, 'objects', 'object_all.png')
    if vis_img in pred_paths:
        pred_paths.remove(vis_img)
    gt_images = load_gt_masks(gt_paths)
    pred_images = load_pred_masks(pred_paths)

    if len(pred_images) == 0 and len(gt_images) == 0:
            precision = 1.0
            precision_per_threshold = [1.0] * len(thresholds)
            tp_fp_fn_tn_per_threshold = np.zeros((len(thresholds), 4))
            tp_fp_fn_tn_per_threshold[:,3] = 1.0

    else:

        if len(pred_images) == 0 and len(gt_images) > 0:
            pred_images = []
            pred_images.append(np.zeros(gt_images[0].shape, np.uint8))

        elif len(pred_images) > 0 and len(gt_images) == 0:
            gt_images = []
            gt_images.append(np.zeros(gt_images[0].shape, np.uint8))

        vis_gt = np.zeros((gt_images[0].shape[0], gt_images[0].shape[1], 3), dtype=np.uint8)
        
        colours = distinctipy.get_colors(len(gt_images))
        for i in range(0, len(gt_images)):
            vis_gt[np.nonzero(gt_images[i])] = np.array(colours[i]) * 255
        save_image(vis_gt, 'vis_gt', output_dir)

        vis_pred = np.zeros((pred_images[0].shape[0], pred_images[0].shape[1], 3), dtype=np.uint8)
        colours = distinctipy.get_colors(len(pred_images))
        for i in range(0, len(pred_images)):
            vis_pred[np.nonzero(pred_images[i])] = np.array(colours[i]) * 255
        save_image(vis_pred, 'vis_pred', output_dir)

        precision, precision_per_threshold, tp_fp_fn_tn_per_threshold = compute_image_precision(gt_images, pred_images, thresholds)
    
    data = dict()
    data['id'] = image_name
    data['precision'] = precision
    for threshold in range(0, len(thresholds)):
        thresh = thresholds[threshold]
        data[f"precision_at_{thresh}"] = precision_per_threshold[threshold]
    return data

#------------------------------------------------------------------------------

def export_stats(path: str, data: dict, thresholds: list):
    '''
    Exports the computed statistics into a CSV file.

            Parameters:
                    `path` (str): export path (directory)
                    `data` (dict): data obtained from evaluate_all_images()
                    `thresholds` (list): list of thresholds used to compute precisions

            Returns:
                    None, writes to a file
    '''
    tresh_fields = [f"precision_at_{i}" for i in thresholds]
    fieldnames = ['id', 'precision'] + tresh_fields
    csv_file = os.path.join(path, 'stats.csv')
    with open(csv_file, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data.keys():
            line = {i:data[entry][i] for i in fieldnames}
            writer.writerow(line)

#------------------------------------------------------------------------------

if __name__ == '__main__':

    # execute only if run as a script
    ap = argparse.ArgumentParser(prog="MSCT Intersection over Union (IoU) computation")
    ap.add_argument('gt_dir', help='Directory containing ground truth object images')
    ap.add_argument('pred_dir', help='Directory containing both predicted object images')
    ap.add_argument('image_name', help='Name of the image to process without extension')

    args = vars(ap.parse_args())
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    data = evaluate_image(
        gt_dir=args['gt_dir'],
        pred_dir=args['pred_dir'],
        image_name=args['image_name'],
        thresholds=thresholds
    )
    
    #export_stats(args['data'], data, thresholds)