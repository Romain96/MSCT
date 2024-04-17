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

from intersection_over_union import evaluate_image

#------------------------------------------------------------------------------

def get_subdirectories(path: str) -> list:
    return [f.name for f in os.scandir(path) if f.is_dir()]

#------------------------------------------------------------------------------

def process_kaggle(input_dir: str, output_dir: str) -> None:

    # thresholds = 50% - 95% (step of 5%)
    thresholds = [i/100 for i in range(50, 100, 5)]
    subdirs = get_subdirectories(input_dir)
    stats = []
    filename = os.path.join(output_dir, 'log.txt')
    image_index = 1

    # computing Intersection over Union
    for subdir in subdirs:
        print(f"Processing image {image_index}/{len(subdirs)} : {subdir}")
        data = evaluate_image(
            gt_dir=input_dir, 
            pred_dir=output_dir, 
            image_name=subdir, 
            thresholds=thresholds, 
            output_dir=os.path.join(output_dir, subdir)
        )
        stats.append(data)
        text = f"{image_index}"
        for key in data.keys():
            text = text + f",{data[key]}"
        if image_index == 1:
            with open(filename, 'w') as log:
                log.write(f"{text}\n")
        else:
            with open(filename, 'a') as log:
                log.write(f"{text}\n")
                    
        image_index += 1
    return stats

#------------------------------------------------------------------------------

def save_stats(stats: list, output_dir: str) -> None:
    filename = os.path.join(output_dir, 'all_stats.csv')
    fieldnames = [i for i in stats[0].keys()]
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for stat in stats:
            writer.writerow(stat)

#------------------------------------------------------------------------------

if __name__ == '__main__':

    # execute only if run as a script
    ap = argparse.ArgumentParser(prog="MSCT Intersection over Union (IoU) evaluation")
    ap.add_argument('indir', help='Input directory (Kaggle Bowl dataset)')
    ap.add_argument('outdir', help='Output directory')
    ap.add_argument('--debug', help='Prints all debug informations (log)', action='store_true')

    args = vars(ap.parse_args())
    stats = process_kaggle(input_dir=args['indir'], output_dir=args['outdir'])
    save_stats(stats, args['outdir'])