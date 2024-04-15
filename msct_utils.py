#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

from os import mkdir
from os.path import join, isdir, isfile
from shutil import rmtree
import cv2 as cv
import numpy as np
from PIL import Image
import tifffile

#------------------------------------------------------------------------------

def load_image(filename: str) -> Image.Image:
    '''
    Loads an image from a file using the given filename.

            Parameters:
                    `filename` (str): path to the image file

            Returns:
                    image (Image): PIL Image
    '''
    image = None

    if isfile(filename):
        image = Image.open(filename)
        if image is None:
            raise Exception(f"The image cannot be loaded from {filename}")
    else:
        raise Exception(f"The given path is invalid : {filename}")
    
    return image

#------------------------------------------------------------------------------

def convert_colour_to_grayscale(image: Image.Image) -> np.ndarray:
    '''
    Converts a loaded image from a colour one to a grayscale one.

            Parameters:
                    `image` (Image): PIL Image

            Returns:
                    grayscale (ndarray): numpy ndarray og the converted grayscale image
    '''
    img = image.convert(mode="L")
    grayscale = np.array(img)
    return grayscale

#------------------------------------------------------------------------------

def save_image(image: np.ndarray, name: str, output_dir='.') -> None:
    '''
    Saves an image to a PNG file.

            Parameters:
                    `image` (ndarray): image as numpy ndarray
                    `name` (str): name of the image file without extension
                    `output_dir` (str): output directory where the file will be saved

            Returns:
                    None
    '''
    filename = join(output_dir, f"{name}.png")
    img = Image.fromarray(image)
    img.save(filename)

#------------------------------------------------------------------------------

def create_dir_remove_if_exist(directory: str) -> None:
    '''
        Creates a directory if it does not already exists.

                Parameters:
                        `directory` (str): path to the directory

                Returns:
                        None
        '''
    if isdir(directory):
        rmtree(directory)
    mkdir(directory)

#------------------------------------------------------------------------------

def normalize_image(image: np.ndarray) -> np.ndarray:
    '''
    Normalizes the signal in range [0,255] for a given image.

            Parameters:
                    `image` (ndarray): image

            Returns:
                    `normalized_image` (ndarray): normalized image, same size as `image`
    '''
    max_value = max(np.amax(image), 1e-6)
    normalized_image = (image / max_value * 255).astype(np.uint8)
    return normalized_image
    
#------------------------------------------------------------------------------
        
def test_if_image_is_colour(image: Image.Image) -> bool:
    '''
    Tests if the input image is a colour image or a grayscale one.

            Parameters:
                    `image` (Image): PIL Image

            Returns:
                    (bool): True if the image is a colour one, False otherwise
    '''
    img = np.array(image)
    if len(img.shape) < 3:
        return False
    elif img.shape[2] == 1:
        return False
    else:
        nb_same = 0
        nb_diff = 0
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                if img.shape[2] > 3:
                    values = [img[i,j,k] for k in range(0, 3)]
                else:
                    values = [img[i,j,k] for k in range(0, img.shape[2])]
                if all(x==values[0] for x in values):
                    nb_same += 1
                else:
                    nb_diff += 1
        if nb_diff == 0:
            return False
        else:
            return True

#------------------------------------------------------------------------------

def load_multi_channel_image(path: str, base_index: int = 0) -> list[np.ndarray]:
    '''
    Loads a multi channel TIF image using Tifffile.
    '''
    image = tifffile.imread(path)
    channel_images = []
    for i in range(0, image.shape[0]):
        if i == base_index:
            #norm = (image[i] / np.max(image[i]) * 255).astype(np.uint8)
            norm = normalize_full_range(image[i]).astype(np.uint8)
        else:
            norm = image[i]
        channel_images.append(norm)
    return channel_images

def normalize_full_range(image: np.ndarray) -> np.ndarray:
    '''
    Normalizes to [0,255] range
    '''
    min_val = np.min(image)
    max_val = np.max(image)
    norm = 255 * (image - min_val) / (max_val - min_val)
    return norm

#------------------------------------------------------------------------------
