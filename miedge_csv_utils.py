#!/usr/bin/env python

__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import os
import shutil
import tifffile
from msct_utils import load_multi_channel_image, create_dir_remove_if_exist

#------------------------------------------------------------------------------

def read_multiple_image_annotations_in_single_csv(path: str) -> dict:
    
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
            data[image][cell_type].add((int(x), int(y)))

    return data

#------------------------------------------------------------------------------

def write_single_image_annotations_csv_as_xy(path: str, data: dict) -> None:
    '''
    Saves annotations for one image as (x, y, type)
    '''
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['x', 'y', 'type'], delimiter=',')
        for key in data.keys():
            coords = data.get(key)
            for x, y in coords:
                writer.writerow({'x': x, 'y': y, 'type': key})

#------------------------------------------------------------------------------

def write_single_image_annotations_csv_as_rc(path: str, data: dict) -> None:
    '''
    Saves annotations for one image as (row, col, type)
    '''
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['row', 'col', 'type'], delimiter=',')
        for key in data.keys():
            coords = data.get(key)
            for x, y in coords:
                writer.writerow({'row': x, 'col': y, 'type': key})

#------------------------------------------------------------------------------

def read_single_image_annotations_csv_as_xy(path: str) -> dict:
    '''
    Loads a single file annotations as (x, y, type)
    '''
    data = dict()
    with open(path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=['x', 'y', 'type'], delimiter=',')
        for row in reader:
            cell_type = row['type']
            if data.get(cell_type) is None:
                data[cell_type] = set()
            data[cell_type].add((int(row['x']), int(row['y'])))
    return data

#------------------------------------------------------------------------------

def read_single_image_annotations_csv_as_rc(path: str) -> dict:
    '''
    Loads a single file annotations as (row, col, type)
    '''
    data = dict()
    with open(path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=['row', 'col', 'type'], delimiter=',')
        for row in reader:
            cell_type = row['type']
            if data.get(cell_type) is None:
                data[cell_type] = set()
            data[cell_type].add((int(row['row']), int(row['col'])))
    return data

#------------------------------------------------------------------------------

def create_image_annotations_from_single_file(input_file: str, output_dir: str):
    '''
    Reads a single CSV file containing annotations for multiple images and exports
    one csv file per image (with the correct name) containing local annotations.
    '''
    if not os.path.exists(input_file):
        Exception(f"Input file {input_file} does not exist !")

    if not os.path.exists(output_dir):
        Exception(f"Output directory {output_dir} does not exist !")

    # read single CSV with all image annotations
    annotations = read_multiple_image_annotations_in_single_csv(path=input_file)

    # create image annotations CSVs
    for image_name in list(annotations.keys()):
        output_file = os.path.join(output_dir, f"{image_name}.csv")
        write_single_image_annotations_csv_as_rc(path=output_file, data=annotations.get(image_name))

#------------------------------------------------------------------------------

def copy_file(input: str, output: str) -> None:
    shutil.copy(input, output)

#------------------------------------------------------------------------------

def move_file(input: str, output: str) -> None:
    shutil.move(input, output)

#------------------------------------------------------------------------------

def find_files_in_dir(input_dir: str, extension: str) -> list[str]:
    files = []
    for file in os.listdir(input_dir):
        if file.endswith(extension):
            files.append(file)
    return files

#------------------------------------------------------------------------------

def find_dirs_in_dir(input_dir: str) -> list[str]:
    dirs = [os.path.basename(f.path) for f in os.scandir(input_dir) if f.is_dir()]
    return dirs
#------------------------------------------------------------------------------

def create_miedge_dataset(input_dir: str, output_dir: str) -> None:
    '''
    Reads the CSV file in `input_dir` and creates one subdirectory in `output_dir`
    for each image in `input_dir`.
    Creates a new CSV file and moves each image of `input_dir` inside 
    the newly-created subdirctories of `output_dir`.
    '''
    if not os.path.exists(input_dir):
        Exception(f"Input directory {input_dir} does not exist !")

    if not os.path.exists(output_dir):
        Exception(f"Output directory {output_dir} does not exist !")

    # find the CSV file and image files
    csv_files = find_files_in_dir(input_dir=input_dir, extension='.csv')
    csv_file = os.path.join(input_dir, csv_files[0])

    # extract image data from the CSV file
    all_annotations = read_multiple_image_annotations_in_single_csv(path=csv_file)

    # for each image (subdirectories) in the input directory
    for image_name in list(all_annotations.keys()):

        # input & output paths
        subdir = os.path.join(output_dir, image_name)
        original_tif = os.path.join(input_dir, f"{image_name}_component_data.tif")
        output_image_path = os.path.join(subdir, 'image.tif')
        output_annotation_path = os.path.join(subdir, 'annotations.csv')

        # creating the output subdirectory (named after the image)
        create_dir_remove_if_exist(subdir)

        # copying the image and extracting its annotations
        copy_file(original_tif, output_image_path)
        annotations = all_annotations.get(image_name)
        write_single_image_annotations_csv_as_rc(path=output_annotation_path, data=annotations)
        
#------------------------------------------------------------------------------

def extract_dapi_from_annotations(data: dict) -> set[tuple[int, int]]:
    '''
    DAPI = sum of all annotations
    '''
    dapi = set()
    for key in data.keys():
        for x, y in data.get(key):
            dapi.add((x, y))
    return dapi

#------------------------------------------------------------------------------

def extract_marker_from_annotations(data: dict, marker: str) -> set[tuple[int, int]]:
    '''
    marker = sum of all keys containing the given marker
    '''
    cells = set()
    for key in data.keys():
        markers = key.split(sep='/')
        process = False
        for candidate in markers:
            if candidate == marker:
                process = True
        if process:
            for x, y in data.get(key):
                cells.add((x, y))
    return cells

#------------------------------------------------------------------------------

def superimpose_annotations_on_image(image: np.ndarray,  annotations: set[tuple[int, int]], output_file: str) -> None:
    plt.cla()
    plt.clf()
    plt.imshow(image)
    for row, col in annotations:
        plt.plot(row, col, 'xr', markersize=5)
    plt.savefig(output_file)

#------------------------------------------------------------------------------

def create_channel_annotation_visualization(input_dir: str, output_dir: str) -> None:
    if not os.path.exists(input_dir):
        Exception(f"Input directory {input_dir} does not exist !")

    if not os.path.exists(output_dir):
        Exception(f"Output directory {output_dir} does not exist !")

    image_dirs = find_dirs_in_dir(input_dir=input_dir)
    for image_dir in image_dirs:

        create_dir_remove_if_exist(os.path.join(output_dir, image_dir))

        # input & output
        sub_image = os.path.join(input_dir, image_dir, 'image.tif')
        sub_annotations = os.path.join(input_dir, image_dir, 'annotations.csv')
        output_subdir = os.path.join(output_dir, image_dir)
        create_dir_remove_if_exist(output_subdir)

        # load image and annotations
        image = load_multi_channel_image(path=sub_image)
        annotations = read_single_image_annotations_csv_as_rc(path=sub_annotations)

        # export DAPI (channel 0)
        dapi_channel = image[0]
        dapi_cells = extract_dapi_from_annotations(annotations)
        dapi_image = os.path.join(output_subdir, 'channel_0_DAPI.png')
        superimpose_annotations_on_image(image=dapi_channel, annotations=dapi_cells, output_file=dapi_image)

        # export single channel markers (GFAP, CD3, CD68, CD34, CD206)
        single_channel_markers = ['astrocyte', 'CD3', 'CD68', 'CD34', 'CD206']
        single_channel_indices = [1, 3, 7, 8, 5]
        for i in range(0, len(single_channel_indices)):
            channel_index = single_channel_indices[i]
            channel_marker = single_channel_markers[i]
            channel_cells = extract_marker_from_annotations(data=annotations, marker=channel_marker)
            channel_image = os.path.join(output_subdir, f"channel_{channel_index}_{channel_marker}.png")
            superimpose_annotations_on_image(image=image[channel_index], annotations=channel_cells, output_file=channel_image)

        # export multi channel markers (IDH1 and ATRX)
        idh1_channel = 4
        atrx_channel = 6
        tumor_marker = 'tumor'
        tumor_cells = extract_marker_from_annotations(data=annotations, marker=tumor_marker)
        idh1_image = os.path.join(output_subdir, f"channel_{idh1_channel}_IDH1_tumor.png")
        atrx_image = os.path.join(output_subdir, f"channel_{atrx_channel}_ATRX_tumor.png")
        superimpose_annotations_on_image(image=image[idh1_channel], annotations=tumor_cells, output_file=idh1_image)
        superimpose_annotations_on_image(image=image[atrx_channel], annotations=tumor_cells, output_file=atrx_image)

        # export other markers
        idh1_channel = 4
        atrx_channel = 6
        other_marker = 'other'
        other_cells = extract_marker_from_annotations(data=annotations, marker=other_marker)
        idh1_image = os.path.join(output_subdir, f"channel_{idh1_channel}_IDH1_other.png")
        atrx_image = os.path.join(output_subdir, f"channel_{atrx_channel}_ATRX_other.png")
        superimpose_annotations_on_image(image=image[idh1_channel], annotations=other_cells, output_file=idh1_image)
        superimpose_annotations_on_image(image=image[atrx_channel], annotations=other_cells, output_file=atrx_image)

#------------------------------------------------------------------------------

if __name__ == "__main__":

    # execute only if run as a script
    ap = argparse.ArgumentParser(prog="MI*EDGE index to image CSV extraction")
    ap.add_argument('indir', help='input directory containing a CSV file and images')
    ap.add_argument('outdir', help="Output directory for image and csv files")
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--create', help='create the dataset from the raw dataset', action='store_true')
    group.add_argument('--visu', help='create visualisation from the dataset', action='store_true')

    args = vars(ap.parse_args())
    if args['create']:
        create_miedge_dataset(
            input_dir=args['indir'], 
            output_dir=args['outdir']
        )
    if args['visu']:
        create_channel_annotation_visualization(
            input_dir=args['indir'], 
            output_dir=args['outdir']
        )

    # Now to be sure, for 916 and 916a, we have
    # DAPI
    # GFAP
    # MS4A4A --> macrophage
    # CD3 --> T-cell; to classify
    # IDH1
    # CD206 --> macrophages
    # ATRX
    # CD68 --> macrophages
    # CD34 --> vessels

    # And for 926, the linkage is
    # DAPI
    # GFAP
    # MS4A4A --> macrophage
    # IDH1
    # CD3 --> T-cell
    # CD206 --> macrophages
    # ATRX
    # CD68 --> macrophages
    # CD34 --> vessels
