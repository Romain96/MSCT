#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"


import numpy as np
from scipy.ndimage import median_filter
import csv
import os
import distinctipy
import cv2 as cv

from mser import MSER
from msct_utils import save_image, normalize_image
from subsampler import Subsampler
from max_tree import MaxTree
from multi_scale_component_tree import MultiScaleComponentTree
from index_to_pixel import Index2Pixel
from pixel_power_two_scale_converter import PixelPower2ScaleConverter
from mser import MSER
from msct_object_divider import MSCTObjectDivider
from msct_miedge_segmentation import identify_single_objects, save_segmentation_as_label_image, save_channel_vectors_as_array

#------------------------------------------------------------------------------

class MSCTMultiChannelWrapper():

    def __init__(self):
        self.base_channel = -1
        self.base_channel_images = []
        self.channel_images = []
        self.annotations = dict()
        self.msct = None
        self.pixel_converter = None
        self.scale_converter = None
        self.augment_index = None
        self.added_rows = 0
        self.added_columns = 0

    def reset(self):
        self.base_channel = -1
        self.base_channel_images = []
        self.channel_images = []
        self.annotations = dict()
        self.msct = None
        self.pixel_converter = None
        self.scale_converter = None
        self.augment_index = None
        self.added_rows = 0
        self.added_columns = 0

    #--------------------------------------------------------------------------

    def get_base_channel(self) -> int:
        '''
        Getter for attribute `base_channel`
        '''
        return self.base_channel

    def set_base_channel(self, base_channel: int) -> None:
        '''
        Setter for attribute `base_channel`
        '''
        self.base_channel = base_channel

    #--------------------------------------------------------------------------

    def get_base_channel_images(self) -> list[np.ndarray]:
        '''
        Getter for attribute `base_channel_images`
        '''
        return self.base_channel_images
    
    def set_base_channel_images(self, base_channel_images: list[np.ndarray]) -> None:
        '''
        Setter for attribute `base_channel_images`
        '''
        self.base_channel_images = base_channel_images

    def get_base_channel_image_at_scale(self, scale: int) -> np.ndarray:
        '''
        Returns the base channel image at the given scale.

                Parameters :
                    `scale` (int): scale at which the image should be retrieved (0 = original)

                Returns : 
                    - image (ndarray): the stored image of the `channel`th image at the `scale` scale
        '''
        if scale < 0 or scale >= len(self.base_channel_images[scale]):
            raise Exception(f"The given scale index {scale} does not exist in the base channel containing {len(self.base_channel_images)} images")
        
        return self.base_channel_images[scale]
    
    #--------------------------------------------------------------------------

    def get_channel_images(self) -> list[np.ndarray]:
        '''
        Getter for attribute `channel_images`
        '''
        return self.channel_images
    
    def set_channel_images(self, channel_images: list[np.ndarray]) -> None:
        '''
        Setter for attribute `channel_images`
        '''
        self.channel_images = channel_images 

    def get_channel_image_at_index(self, channel: int, scale: int) -> np.ndarray:
        '''
        Returns the channel image at the given index.

                Parameters :
                    `channel` (int): index at which the image should be retrieved

                Returns : 
                    - image (ndarray): the stored image of the `channel`th image
        '''
        if channel < 0 or channel >= len(self.channel_images):
            raise Exception(f"The given channel index {channel} does not exist in the multi channel image containing {len(self.channel_images)} images")
        
        return self.channel_images[channel]

    #--------------------------------------------------------------------------

    def get_annotations(self) -> dict:
        '''
        Getter for attribute `annotations`
        '''
        return self.annotations
    
    def set_annotations(self, annotations: dict) -> None:
        '''
        Setter for attribute `annotations`
        '''
        self.annotations = annotations

    #--------------------------------------------------------------------------

    def get_msct(self) -> MultiScaleComponentTree:
        '''
        Getter for attribute `msct`
        '''
        return self.msct

    def set_msct(self, msct: MultiScaleComponentTree) -> None:
        '''
        Setter for attribute `msct`
        '''
        self.msct = msct

    #--------------------------------------------------------------------------

    def get_pixel_converter(self) -> Index2Pixel:
        '''
        Getter for attribute `pixel_converter`
        '''
        return self.pixel_converter

    def set_pixel_converter(self, pixel_converter: Index2Pixel) -> None:
        '''
        Setter for attribute `pixel_converter`
        '''
        self.pixel_converter = pixel_converter

    #--------------------------------------------------------------------------

    def get_scale_converter(self) -> PixelPower2ScaleConverter:
        '''
        Getter for attribute `scale_converter`
        '''
        return self.scale_converter

    def set_scale_converter(self, scale_converter: PixelPower2ScaleConverter) -> None:
        '''
        Setter for attribute `scale_converter`
        '''
        self.scale_converter = scale_converter

    #--------------------------------------------------------------------------

    def get_index(self) -> int:
        '''
        Getter for attribute `index`
        '''
        return self.index
    
    def set_index(self, index: int) -> None:
        '''
        Setter for attribute `index`
        '''
        self.index = index

    #--------------------------------------------------------------------------

    def get_added_rows(self) -> int:
        '''
        Getter for attribute `added_rows`
        '''
        return self.added_rows
    
    def set_added_rows(self, added_rows: int) -> None:
        '''
        Setter for attribute `added_rows`
        '''
        self.added_rows = added_rows
    
    #--------------------------------------------------------------------------

    def get_added_columns(self) -> int:
        '''
        Getter for attribute `added_columns`
        '''
        return self.added_columns
    
    def set_added_columns(self, added_columns: int) -> None:
        '''
        Setter for attribute `added_columns`
        '''
        self.added_columns = added_columns

    #--------------------------------------------------------------------------

    def apply_median_filter(self, image: np.ndarray, size: int) -> np.ndarray:
        '''
        Applies a median filter of size `size`x`size` to the layer image `image`.

                Parameters:
                        `image` (ndarray): layer image grayscale
                        `size` (int): median filter size
                
                Returns:
                        `filtered` (ndarray): filtered grayscale image
        '''
        filtered = median_filter(image, size=(size,size))
        return filtered

    #--------------------------------------------------------------------------

    def fix_image_dimensions_for_subdivision(self, image: np.ndarray, n: int) -> np.ndarray:
        '''
        Fixes the image dimensions so that they are divisible by 2^n by adding empty rows
        and columns at the end of it. These added rows/columns have no influence on processing
        as the maximum value will always be taken when performing subdivisions and will be
        removed when the reconstruction step occurs, insuring that the reconstructing image
        matches the original image dimensions. Fills both `added_rows` and `added_columns` attributes.

                Parameters:
                        `image` (ndarray): original gray-level image
                        `n` (int): number of subdivision steps

                Returns:
                        `fixed` (ndarray): same as `image` but with added empty rows/columns
        '''
        width = image.shape[1]
        height = image.shape[0]

        # if width not divisible by 2^n
        if width % pow(2, n) != 0:
            cols = 1
            w = width + cols
            while w % pow(2, n) != 0:
                cols += 1
                w = width + cols
            self.added_columns = cols
        else:
            self.added_columns = 0

        # if height not divisible by 2^n
        if height % pow(2, n) != 0:
            rows = 1
            h = height + rows
            while h % pow(2, n) != 0:
                rows += 1
                h = height + rows
            self.added_rows = rows
        else:
            self.added_rows = 0

        if self.get_added_rows() != 0 or self.get_added_columns() != 0:
            fixed = np.zeros((height + self.get_added_rows(), width + self.get_added_columns()), dtype=image.dtype)
            fixed[0:height, 0:width] = image
            return fixed
        else:
            return image

    #--------------------------------------------------------------------------

    def apply_CLAHE(self, image: np.ndarray) -> np.ndarray:
        '''
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) 
        on a grayscale image using the OpenCV implementation.

                Parameters:
                        `image` (nadarray): grayscale image

                Returns:
                        `enhanced` (ndarray): grayscale image `image` after CLAHE applied
        '''
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
        enhanced = clahe.apply(image)
        return enhanced

    #--------------------------------------------------------------------------

    def downsample_image(
        self, 
        channel_images: list[np.ndarray], base_channel: int, 
        n: int, method='maximum', 
        use_median_filter=True, median_filter_size=3, 
        invert_image=False, normalize=False
    ) -> None:
        '''
        Downsamples the given image n times using the given subsampling method (maximum, mean, minimum)

                Parameters:
                        `channel_images` (ndarray): list of channel images
                        `base_channel` (int): index of the base channel from which the MSCT will be computed
                        `n` (int): number of successive subsampling steps
                        `method` (str): subsampling method ('maximium', 'minimum' or 'mean')
                        `use_median_filter` (bool): whether to apply a median filter to the image before processing it
                        `median_filter_size` (int): size of the median filter, only relevant if `use_median_filter` is True
                        `invert_image` (bool): whether to invert image values (black on white instead as white on black)
                        `normalize` (bool): whether to normalize images or load them as is

                Returns: none
        '''
        for i in range(0, len(channel_images)):

            if invert_image:
                channel_images[i] = 255 - channel_images[i]
            if normalize:
                channel_images[i] = normalize_image(channel_images[i])
            channel_images[i] = self.fix_image_dimensions_for_subdivision(channel_images[i], n)

            if i == base_channel: 
                if use_median_filter:
                    channel_images[i] = normalize_image(image=self.apply_median_filter(channel_images[i], median_filter_size))
                #channel_images[i][np.where(channel_images[i] < round(5*255/100))] = 0
                channel_images[i] = self.apply_CLAHE(image=channel_images[i])

        # processing the base_channel with downsampling
        base_channel_image = channel_images[base_channel]
        base_data = [base_channel_image]

        # subsampling n times by a factor of 2 using the required method
        previous_image = base_channel_image
        current_image = base_channel_image
        for i in range(0, n):
            if method == 'maximum':
                current_image = Subsampler.subsample_image_maximum(previous_image)
            elif method == 'minimum':
                current_image = Subsampler.subsample_image_minimum(previous_image)
            elif method == 'mean':
                current_image = Subsampler.subsample_image_mean(previous_image)
            
            base_data.append(current_image)
            previous_image = current_image

        # initialize a pixel and a scale converter
        pixel_converter = Index2Pixel(current_image.shape[1]) # smallest image
        scale_converter = PixelPower2ScaleConverter()
        self.set_pixel_converter(pixel_converter)
        self.set_scale_converter(scale_converter)
        self.set_index(n)
        self.set_base_channel(base_channel)
        self.set_base_channel_images(base_data)
        self.set_channel_images(channel_images)

    #--------------------------------------------------------------------------

    def build_base_msct(self, inverted_image=False) -> MultiScaleComponentTree:
        '''
        Builds a MSCT with the lowest scale image

                Parameters:
                        `inverted_image` (bool): whether the image has to be inverted

                Returns:
                        None
        '''
        # building a max-tree on the lowest sampled image
        ct = MaxTree()
        image = self.get_base_channel_image_at_scale(self.get_index())
        ct.build_component_tree(image, inverted_image)

        # initializing a MSCT from the max-tree
        msct = MultiScaleComponentTree()
        msct.init_from_component_tree(ct, self.get_pixel_converter(), self.get_index())
        self.set_msct(msct)

    #--------------------------------------------------------------------------

    def compute_mser(self, delta: int) -> None:
        '''
        Computes MSER stabilities for all nodes of the MSCT.

                Parameters:
                        `delta` (int): gray-level delta

                Returns:
                        None
        '''
        msct = self.get_msct()
        root = msct.get_root()
        msct.compute_mser(node=root, delta=delta)

    #--------------------------------------------------------------------------

    def compute_mser_percent_height(self, percent_height: float) -> None:
        '''
        Computes MSER stabilities for all nodes of the MSCT using delta as a percentage of the tree height.

                Parameters:
                        `percent_height` (float): percentage of the tree height used as delta in [0,1]

                Returns:
                        None
        '''
        msct = self.get_msct()
        root = msct.get_root()
        height = msct.compute_max_tree_height()
        delta = max(1, int(round(height * percent_height)))
        msct.compute_mser(node=root, delta=delta)

    #--------------------------------------------------------------------------

    def augment_msct_mser(self, max_area: int, max_mser: float, invert_image=False) -> None:
        '''
        Augments a set of nodes with one higher scale image using MSER stabilities.
        MSER should be computed prior to running this method !

                Parameters:
                        `max_area` (int): maximum area for a flat zone to be considered a valid candidate for enrichment
                        `max_mser` (float): maximum MSER value for a flat zone to be considered a valid candidate for enrichment
                        `invert_image` (bool): whether to invert the image

                Returns:
                        None
        '''
        msct = self.get_msct()
        current_scale = self.get_index()
        augment_scale = current_scale - 1
        augment_image = self.get_base_channel_image_at_scale(augment_scale)

        filtered_minima = MSER.compute_mser_candidates(msct=msct, scale=current_scale, max_area=max_area, max_mser=max_mser)

        for node in filtered_minima:
            msct.augment_node_sequential(
                node=node, target_scale=augment_scale, 
                scale_image=augment_image, 
                pixel_converter=Index2Pixel(augment_image.shape[1]),
                invert_image=invert_image
            )
        self.set_index(augment_scale)

    #--------------------------------------------------------------------------

    def populate_histograms_with_channel_images(self) -> None:
        '''
        Recursively populates each MSCT node with a histogram of all channel images excluding the base_channel one
        '''
        #TODO MSCT METHOD
        channel_images = [self.get_channel_images()[i] for i in range(0, len(self.get_channel_images())) if i != self.get_base_channel()]
        print(len(channel_images))
        self.get_msct().build_channel_histogram(channel_images)

    #--------------------------------------------------------------------------

    def save_channel_images(self, output_dir: str) -> None:
        '''
        Saves each channel image to the given output directory.

                Parameters :
                        `output_dir` (str): path to the output directory

                Returns : none, writes images on disk
        '''
        if not os.path.exists(output_dir):
            Exception(f"Cannot save channel images to {output_dir}, directory does not exist")

        channel_images = self.get_channel_images()
        for index in range(0, len(channel_images)):
            channel_image = (channel_images[index] / np.max(channel_images[index]) * 255).astype(np.uint8)
            save_image(channel_image, f"channel_{index}", output_dir)

    #--------------------------------------------------------------------------

    def load_miedge_annotations(self, filepath: str) -> None:
        '''
        Loads MI*EDGE dot annotations for the current image from a CSV file.

                Parameters :
                        `filepath` (str): path to the CSV file containing dot annotations

                Returns : none, assigns `annotations`
        '''

        with open(filepath, 'r') as csv_file:

            # x , y , cell type
            data = dict()
            reader = csv.reader(csv_file, delimiter=';')
            for row in reader:
                x = row[0]
                y = row[1]
                key = row[2]
                if data.get(key) is None:
                    data[key] = set()
                data[key].add((x, y))
            self.set_annotations(data)

    #--------------------------------------------------------------------------

    def divide_objects(
        self, max_mser: float, max_area_factor: float, min_area: int) -> tuple[list[np.ndarray], list[list[int]]]:
        '''
        Attempts to identify single objects (nodes) among maximally augmented nodes.

                Parameters:
                        `max_mser` (float): Maximum MSER value for objects
                        `max_area_factor` (float): maximum area for chosen nodes
                        `min_area` (int): minimum area to consider a flatzone as a nucleus

                Returns:
                        `all_objects` (list): list of segmented nuclei
                        `channel_vectors` (list): list of channel vectors for each nucleus
        '''
        
        channel_images = self.get_channel_images()
        channels_without_base = []
        for i in range(0, len(channel_images)):
            if i != self.get_base_channel():
                channels_without_base.append(channel_images[i])
        all_objects, channel_vectors = identify_single_objects(
            msct=self.get_msct(), 
            scale=self.get_index(), 
            image=self.get_base_channel_images()[0], 
            max_mser=max_mser, 
            max_area_factor=max_area_factor,
            min_area=min_area,
            annotations=self.get_annotations(),
            channel_images=channels_without_base
        )

        return (all_objects, channel_vectors)
    
    #--------------------------------------------------------------------------

    def export_objects_as_images(self, images: list[np.ndarray], output_dir: str, all: bool) -> None:
        '''
        Exports all identified objects as binary images.

                Parameters:
                        `images` (list): list of clusters being boolean images
                        `output_dir` (str): output directory path
                        `all` (bool): whether to save all individual objects as images or just a combined colour image

                Returns:
                        None
        '''
        if not os.path.isdir(output_dir):
            Exception(f"{output_dir} is not a directory")
        
        image = self.get_base_channel_images()[0]
        h = image.shape[0] - self.get_added_rows()
        w = image.shape[1] - self.get_added_columns()
        all_objects_image = np.zeros((image.shape[0], image.shape[1], 3), image.dtype)
        colours = distinctipy.get_colors(len(images), n_attempts=50)
        index = 0
        for obj in images:
            fz_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            colour = colours[index]
            coords = np.where(obj > 0)
            fz_image[coords] = 255          
            all_objects_image[coords] = np.array(colour) * 255
            if all:
                filename = f"object_{index}"
                fz_image = fz_image[0:h, 0:w]
                save_image(image=fz_image, name=filename, output_dir=output_dir)
            index += 1
        all_objects_image = all_objects_image[0:h, 0:w]
        save_image(image=all_objects_image, name='object_all', output_dir=output_dir)

    #--------------------------------------------------------------------------

    def export_objects_as_label_image(self, images: list[np.ndarray], output_dir: str) -> None:
        '''
        Exports all identified objects as a single integer label image.

                Parameters:
                        `images` (list): list of binary segmented nuclei
                        `output_dir` (str): output directory path

                Returns:
                        None
        '''
        ref = self.get_base_channel_images()[0]
        save_segmentation_as_label_image(data=images, ref_image=ref, output_dir=output_dir, output_name='labels')

    #--------------------------------------------------------------------------

    def export_channel_vectors_as_array(self, vectors: list[list[int]], output_dir: str) -> None:
        '''
        Exports all channel vectors as a single array.

                Parameters:
                        `channel_vectors` (list): list of channel vectors for each nucleus
                        `output_dir` (str): output directory path

                Returns:
                        None
        '''
        save_channel_vectors_as_array(data=vectors, output_dir=output_dir, output_name='vectors')

    #--------------------------------------------------------------------------

    def save_msct_dot(self, filename: str) -> None:
        '''
        Interface for MultiScaleComponentTree.save_dot() method.

                Parameters:
                        `filename` (str): filename where the MSCT is to be saved without extension

                Returns:
                        None
        '''
        msct = self.get_msct()
        msct.save_dot(filename=filename)

    #--------------------------------------------------------------------------

    def save_flat_zones(self, output_dir: str, fz_prefix: str, binary: bool) -> None:
        '''
        Saves all flat zones (one per node) of the MSCT names after their node's ID.

                Parameters:
                        `output_dir` (str): output directory for saving
                        `fz_prefix` (str): flat zones prefix name
                        `binary` (bool): whether to save the flat zone as a binary zone or to assign it its original gray level

                Returns:
                        None
        '''
        msct = self.get_msct()
        nodes = msct.get_nodes()
        image = self.get_base_channel_image_at_scale(0)
        for node in nodes:
            fz = msct.reconstruct_flat_zone(node, image, binary=binary)
            fz_name = f"{fz_prefix}_{node.get_id()}"
            save_image(image=fz, name=fz_name, output_dir=output_dir)

    #--------------------------------------------------------------------------

    def reconstruct_augmented_nodes(self, output_dir: str, fz_prefix: str, binary: bool) -> None:
        '''
        Reconstructs only augmented nodes.

                Parameters:
                        `output_dir` (str): output directory for saving
                        `fz_prefix` (str): flat zones prefix name
                        `binary` (bool): whether to save the flat zone as a binary zone or to assign it its original gray level

                Returns:
                        None
        '''
        msct = self.get_msct()
        augmented_nodes = msct.get_augmented_nodes()
        image = self.get_base_channel_image_at_scale(0)

        for node in augmented_nodes:
            aug_fz = msct.reconstruct_flat_zone(node, image, binary=binary)
            aug_fz_name = f"{fz_prefix}_{node.get_id()}"
            save_image(image=aug_fz, name=aug_fz_name, output_dir=output_dir)

    #--------------------------------------------------------------------------

    def reconstruct_all_augmented_nodes(self, output_dir: str, binary: bool) -> None:
        '''
        Reconstructs only augmented nodes.

                Parameters:
                        `output_dir` (str): output directory for saving
                        `binary` (bool): whether to save the flat zone as a binary zone or to assign it its original gray level

                Returns:
                        None
        '''
        msct = self.get_msct()
        image = self.get_base_channel_image_at_scale(0)
        rec_image = msct.reconstruct_image_augment_nodes_only(image, binary=binary)
        save_image(image=rec_image, name='augmented_nodes', output_dir=output_dir)

    #--------------------------------------------------------------------------

    def save_subsampled_images(self, subsampled_dir: str, subsampled_name: str) -> None:
        '''
        Saves the subsampled images for each scale.

                Parameters:
                        `subsampled_dir` (str): output directory for saving subsampled images
                        `subsampled_name` (str): subsampled image name
        '''
        data = self.get_base_channel_images()
        for index in range(0, len(data)):
            image = data[index]
            scale = pow(2, 2*index)
            out_name = f"{subsampled_name}_{scale}"
            save_image(image=image, name=out_name, output_dir=subsampled_dir)

    #--------------------------------------------------------------------------

    def reconstruct_image(self, reconstructed_dir: str, reconstructed_name: str, normalize: bool) -> None:
        '''
        Reconstructs the image at the global scale using its current MSCT.

                Parameters:
                        `reconstructed_dir` (str): output directory for saving reconstructed images
                        `reconstructed_name` (str): reconstructed image name
                        `normalize` (bool): whether to normalize the image after reconstruction
        '''
        msct = self.get_msct()
        layer_image = self.get_base_channel_images()[0]
        reconstructed = msct.reconstruct_image(layer_image)
        if normalize:
            reconstructed = self.normalize_image(reconstructed)
        h = reconstructed.shape[0] - self.get_added_rows()
        w = reconstructed.shape[1] - self.get_added_columns()
        image = reconstructed[0:h, 0:w]
        save_image(image=image, name=reconstructed_name, output_dir=reconstructed_dir)

    #--------------------------------------------------------------------------

    def normalize_image(self, image: np.ndarray):
        '''
        Normalizes the signal in range [0,255] for a  image.

                Parameters:
                        `image` (ndarray): grayscale image

                Returns:
                        `normalized_image` (ndarray): normalized `image`
        '''
        normalized_image = np.zeros(image.shape, dtype=image.dtype)
        amax = np.amax(image)
        if amax == 0:
            normalized_image = image
        else:
            normalized_image = image / amax * 255
        return normalized_image.astype(np.uint8)
