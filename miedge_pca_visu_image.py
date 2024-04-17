#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import distinctipy
import csv
from miedge_csv_utils import read_single_image_annotations_csv_as_rc

#------------------------------------------------------------------------------

def show_pca_classes(label_image: np.ndarray, annotations: dict, vectors: np.ndarray) -> np.ndarray:

    # get all labels in the annotations
    all_labels = list(set(annotations.values()))
    colors = distinctipy.get_colors(len(all_labels))

    fig, ax = plt.subplots(1, 2)
    
    # using PCA to reduce 8D data to 2D data
    pca = PCA(n_components=2) # 2D data
    reduced = pca.fit_transform(vectors)
    
    # showing original labels on reduced dots
    
    ax[0].set_title("PCA + labels")
    ax[1].set_title("Dots + labels (GT)")
    for dot in list(annotations.keys()):
        label_of_dot = annotations.get(dot)
        label_of_dot_in_image = label_image[dot[1],dot[0]]
        if label_of_dot_in_image > 0:
            rdot = reduced[label_of_dot_in_image-1]
            color_index = all_labels.index(label_of_dot)
            color = colors[color_index]
            ax[0].scatter(rdot[0], rdot[1], marker='.', color=color, label=label_of_dot)
            ax[1].scatter(dot[1], dot[0], marker='.', color=color, label=label_of_dot)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    #plt.savefig(f"pca_dots.png")
    plt.show()
    return reduced

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

def load_label_image(input_dir: str) -> np.ndarray:
    infile = os.path.join(input_dir, 'labels.npy')
    label_img = np.load(infile)
    return label_img

#------------------------------------------------------------------------------

def load_vectors(input_dir: str) -> np.ndarray:
    infile = os.path.join(input_dir, 'vectors.npy')
    vectors = np.load(infile)
    return vectors

#------------------------------------------------------------------------------

if __name__ == '__main__':

    # execute only if run as a script
    ap = argparse.ArgumentParser(prog="MIEDGE histogram PCA")
    ap.add_argument('annotations', help='path the a CSV file containing the image ground truth annotations')
    ap.add_argument('indir', help='Input directory containing the label image and vectors (numpy files)')

    args = vars(ap.parse_args())
    annotation_path = args['annotations']
    input_dir = args['indir']

    # loading data
    annotations = invert_gt(read_single_image_annotations_csv_as_rc(annotation_path))
    label_image = load_label_image(input_dir=input_dir)
    vectors = load_vectors(input_dir=input_dir)

    show_pca_classes(label_image=label_image, annotations=annotations, vectors=vectors)

