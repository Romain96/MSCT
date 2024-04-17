#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import argparse
import os
import numpy as np
import distinctipy

from msct_utils import save_image

#------------------------------------------------------------------------------

def load_label_image(infile: str) -> np.ndarray:
    print(f"Loading label image as numpy in {infile}")
    label_image = np.load(infile, allow_pickle=True)
    print(f"shape {label_image.shape}")
    return label_image

#------------------------------------------------------------------------------

def label_to_color(label_image: np.ndarray) -> np.ndarray:
    '''
    Converts a label image (each CC has one unique index) to a colour image
    with each label/index being assigned to a unique RGB colour
    '''
    color_image = np.zeros((label_image.shape[0], label_image.shape[1], 3), dtype=np.uint8)
    nb_labels = np.max(label_image.flatten())
    print(f"{nb_labels} labels found")
    colors = distinctipy.get_colors(nb_labels, n_attempts=50)
    for label_index in range(0, nb_labels):
        label = label_index + 1
        color_image[np.where(label_image == label)] = np.array(colors[label_index]) * 255
    return color_image

#------------------------------------------------------------------------------

def convert_label_image_to_color_image(infile: str, output_dir: str, output_name: str) -> None:
    label_image = load_label_image(infile=infile)
    color_image = label_to_color(label_image=label_image)
    save_image(image=color_image, name=output_name, output_dir=output_dir)

#------------------------------------------------------------------------------

if __name__ == "__main__":

    # execute only if run as a script
    ap = argparse.ArgumentParser(prog="MI*EDGE tumor classification")
    ap.add_argument('infile', help='input npy label image file')
    ap.add_argument('outdir', help="Output directory")
    ap.add_argument('outname', help="Output name")

    args = vars(ap.parse_args())
    convert_label_image_to_color_image(
        infile=args['infile'], 
        output_dir=args['outdir'], 
        output_name=args['outname']
    )