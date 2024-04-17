#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import argparse
import os
from miedge_label_image_to_color_image import convert_label_image_to_color_image
from miedge_process import find_dirs_in_dir
from msct_utils import create_dir_remove_if_exist

#------------------------------------------------------------------------------

def process_dataset(input_dir: str, output_dir: str) -> None:
    create_dir_remove_if_exist(directory=output_dir)
    subdirs = find_dirs_in_dir(input_dir=input_dir)
    for subdir in subdirs:
        label_image_path = os.path.join(input_dir, subdir, 'labels.npy')
        convert_label_image_to_color_image(
            infile=label_image_path, 
            output_dir=output_dir, 
            output_name=subdir
        )

#------------------------------------------------------------------------------

if __name__ == "__main__":

    # execute only if run as a script
    ap = argparse.ArgumentParser(prog="MI*EDGE tumor classification")
    ap.add_argument('indir', help='Input directory with label images in subdirectories')
    ap.add_argument('outdir', help="Output directory to save color images")

    args = vars(ap.parse_args())
    process_dataset(input_dir=args['indir'], output_dir=args['outdir'])