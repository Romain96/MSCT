#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import argparse
import os
import numpy as np
import csv

from miedge_csv_utils import find_dirs_in_dir

#------------------------------------------------------------------------------

def read_all_gt_annotations_as_rc(path: str) -> dict:
    
    if not os.path.isfile(path):
        Exception(f"CSV file {path} does not exist !")

    data = dict()
    # -> image name ; [ignored] ; x ; y ; label 
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        # ignoring the header
        next(reader, None)
        for row in reader:
            image, cell_id, x, y, cell_type = row
            if data.get(image) is None:
                data[image] = dict()
            if data[image].get(cell_type) is None:
                data[image][cell_type] = set()
            data[image][cell_type].add((int(y), int(x)))

    return data

#------------------------------------------------------------------------------

def invert_gt(dots: dict) -> dict:
    '''
    Dict of labels, pixels to pixels, labels
    '''
    inverted = dict()
    for label in list(dots.keys()):

        for row, col in dots.get(label):
            inverted[(int(row), int(col))] = label
    return inverted

#------------------------------------------------------------------------------

def eval_segmentation_image(nuclei: np.ndarray, annotations: dict) -> tuple[int, int, int, float, float]:
    '''
    Evaluates the segmentated nuclei in regards to ground truth dot annotations.

            Parameters:
                    `nuclei` (ndarray): ndarray representing each nucleus as a connected component
                    `annotations` (dict): dictionnary of key = (row,col) pixel coordinates and value = class label
            
            Returns:
                    `tp` (int): the number of True Positive segmented nuclei
                    `fp` (int): the number of False Positive segmented nuclei
                    `fn` (int): the number of False Negative segmented nuclei
                    `precision` (float): the precision of segmented nuclei
                    `recall` (float): the recall of segmented nulei
    '''
    tp = 0
    fp = 0
    fn = 0

    # for each label in nuclei keep track of whether or not it has been matched with a dot annotation
    dot_matching = [False] *  (np.max(nuclei) + 1)
    all_dots = list(annotations.keys())

    # for all dot, 
    # if the image is > 0 in the coordinate then
    #   either dot_matching is -1 and the dot is matched with this nucleus --> true positive 
    #   or dot_matching != -1 and the nucleus is already matched --> false negative (not segmented or detected)
    # else 
    #   no data --> false negative (not detected)
    for dot in all_dots:
        label = nuclei[dot]
        if label > 0:
            if not dot_matching[label]:
                tp += 1
                dot_matching[label] = True
            else:
                fn += 1
        else:
            fn += 1
    # false positives = unmatched labels
    fp = len(dot_matching) - np.count_nonzero(dot_matching)

    # precision = TP / (TP + FP)
    if tp + fp == 0:
        precision = 1.0
    else:
        precision = tp / (tp + fp)
    # recall = TP / (TP + FN)
    if tp + fn == 0:
        recall = 1.0
    else:
        recall = tp / (tp + fn)

    return (tp, fp, fn, precision, recall)

#------------------------------------------------------------------------------

def save_dataset_stats(stats: dict) -> None:
    with open('dataset_seg_stats.csv', 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['image', 'tp', 'fp', 'fn', 'precision', 'recall'], delimiter=',')
        writer.writeheader()
        for image_name in list(stats.keys()):
            tp, fp, fn, precision, recall = stats[image_name]
            writer.writerow({
                'image': image_name,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': precision,
                'recall': recall
            }
        )

#------------------------------------------------------------------------------

def eval_segmentation_dataset(gt_dots_path: str, input_dir: str) -> dict:
    # load gt annotations
    gt_dots = read_all_gt_annotations_as_rc(path=gt_dots_path)
    # get all image subdirs
    image_dirs = find_dirs_in_dir(input_dir=input_dir)
    d_tp = 0
    d_fp = 0
    d_fn = 0
    d_precision = 0.0
    d_recall = 0.0
    image_stats = dict()
    for image_dir in image_dirs:
        label_image_path = os.path.join(input_dir, image_dir)
        label_image = load_label_image(input_dir=label_image_path)
        image_annotations = invert_gt(gt_dots.get(image_dir))
        tp, fp, fn, precision, recall = eval_segmentation_image(nuclei=label_image, annotations=image_annotations)
        print(image_dir, tp, fp, fn, precision, recall)
        image_stats[image_dir] = (tp, fp, fn, precision, recall)
        d_tp += tp
        d_fp += fp
        d_fn += fn

    if d_tp + d_fp == 0:
        d_precision = 1.0
    else:
        d_precision = d_tp / (d_tp + d_fp)

    if d_tp + d_fn == 0:
        d_recall = 1.0
    else:
        d_recall = d_tp / (d_tp + d_fn)

    save_dataset_stats(image_stats)

    return (d_tp, d_fp, d_fn, d_precision, d_recall)

#------------------------------------------------------------------------------

def load_label_image(input_dir: str) -> np.ndarray:
    infile = os.path.join(input_dir, 'labels.npy')
    labels = np.load(infile)
    return labels

#------------------------------------------------------------------------------

if __name__ == '__main__':

    # execute only if run as a script
    ap = argparse.ArgumentParser(prog="MIEDGE histogram PCA")
    ap.add_argument('gt', help='ground truth ')
    ap.add_argument('indir', help="input dir with segmented dots")

    args = vars(ap.parse_args())
    gt_path = args['gt']
    indir = args['indir']

    tp, fp, fn, precision, recall = eval_segmentation_dataset(gt_dots_path=gt_path, input_dir=indir)
    print(f"TP {tp} FP {fp} FN {fn} precision {precision} recall {recall}")
