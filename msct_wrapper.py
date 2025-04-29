#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"


import numpy as np
from scipy.ndimage import generate_binary_structure, grey_erosion, grey_dilation, median_filter
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from mser import MSER
from msct_utils import save_image, normalize_image
from subsampler import Subsampler
from max_tree import MaxTree
from multi_scale_component_tree import MultiScaleComponentTree
from index_to_pixel import Index2Pixel
from pixel_power_two_scale_converter import PixelPower2ScaleConverter
from mser import MSER
from msct_object_divider import MSCTObjectDivider
from msct_segmentation import identify_single_objects

#------------------------------------------------------------------------------

class MSCTWrapper():
	"""
	This class provides a wrapper to store a grayscale image alongside its MSCT.

	Attributes
	----------
	image : ndarray
		A grayscale image (as Numpy array).
	data: dict
		The scaled images (downsampled from `image`).
	msct: MultiScaleComponentTree
		The MSCT of the stored image.
	pixel_converted: Index2Pixel
		A 1D index to 2D pixel converter.
	scale_converted: PixelPower2ScaleConverter
		A pixel scale converter.
	added_rows: int
		The number of added rows to the `image`.
	added_columns: int
		The number of added columns to the `image`.
	
	Methods
	-------
	apply_median_filter(ndarray, int)
		Applies a median filter to a grayscale image.
	subsample_image(ndarray, int, str, bool, int, bool)
		Downsamples a grayscale image.
	fix_image_dimensions_for_subdivision(ndarray, int)
		Fixes a grayscale image for downsampling by adding rows and/ot columns to it.
	build_base_msct(bool): MultiScaleComponentTree
		Builds the MSCT of the stored image.
	compute_mser(delta: int)
		Computes the MSER stability values of the MSCT nodes with delta as a flat value.
	compute_mser_percent_height(percent_height: float)
		Computes the MSER stability values of MSCT nodes with delta as a percentage of the MSCT height.
	augment_msct_mser(int, float, bool)
		Augments the MSCT based on the MSER values of its nodes.
	divide_objects(float, int)
		Divides the MSCT nodes to obtain single objects segmentation.
	export_objects_as_images(list[ndarray], str, bool)
		Exports all segmented objects as images.
	export_objects_for_kaggle(list[ndarray])
		Exports all segmented objets for Kaggle challenge.
	save_msct_dot(str, bool)
		Saves the MSCT using the DOT format.
	save_flat_zones(str, str, bool)
		Saves all flat zones as images.
	reconstruct_augmented_nodes(str, str, bool)
		Reconstructs an augmented nodes and saves it as an image.
	reconstruct_all_augmented_nodes(str, bool)
		Reconstruct all augmented nodes and saves them as images.
	save_subsampled_images(str, str)
		Saves all downsampled images.
	reconstruct_image(str, str, bool)
		Reconstructs the image from its MSCT and saves it.
	normalize_image(ndarray)
		Normalizes a grayscale image.
	
	Written by Romain PERRIN (<romain.perrin@unistra.fr>).
	"""

	def __init__(self):
		"""
		Initializes all attributes.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.image = None
		self.data = dict()
		self.msct = None
		self.pixel_converter = None
		self.scale_converter = None
		self.augment_index = None
		self.added_rows = 0
		self.added_columns = 0

	#--------------------------------------------------------------------------

	def reset(self):
		"""
		Resets all attributes.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.image = None
		self.data = dict()
		self.msct = None
		self.pixel_converter = None
		self.scale_converter = None
		self.augment_index = None
		self.added_rows = 0
		self.added_columns = 0

	#--------------------------------------------------------------------------

	def get_image(self) -> np.ndarray:
		"""
		Getter for attribute **image**.

		Returns
		-------
		: ndarray
			The stored grayscale image.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.image
	
	#--------------------------------------------------------------------------

	def set_image(self, image: np.ndarray) -> None:
		"""
		Setter for attribute **image**.

		Parameters
		----------
		image: ndarray
			A grayscale image.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""
		self.image = image

	#--------------------------------------------------------------------------

	def get_data(self) -> dict:
		"""
		Getter for attribute **data**.

		Returns
		-------
		: dict
			The dictionary containing downsampled images.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.data
	
	#--------------------------------------------------------------------------
	
	def set_data(self, data: dict) -> None:
		"""
		Setter for attribute **data**.

		Parameters
		----------
		data : dict
			A dictionary containing downsampled images.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.data = data

	#--------------------------------------------------------------------------

	def get_scale_at_index(self, index: int) -> str:
		"""
		Returns the scale at **index**.

		Parameters
		----------
		index: int
			The desired index.

		Returns
		-------
		: str
			The scale corresponding to the given index.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.get_data()[index]['scale']
	
	#--------------------------------------------------------------------------
	
	def get_image_at_index(self, index: int) -> str:
		"""
		Returns the image at **index**.

		Parameters
		----------
		index: int
			The desired index.

		Returns
		-------
		: ndarray
			The image stored for the given index.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.get_data()[index]['image']

	#--------------------------------------------------------------------------

	def get_msct(self) -> MultiScaleComponentTree:
		"""
		Getter for attribute **msct**.

		Returns
		-------
		: MultiScaleComponentTree
			The stored MSCT.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.msct
	
	#--------------------------------------------------------------------------

	def set_msct(self, msct: MultiScaleComponentTree) -> None:
		"""
		Setter for attribute **msct**.

		Parameters
		----------
		msct: MultiScaleComponentTree
			The MSCT to store.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.msct = msct

	#--------------------------------------------------------------------------

	def get_pixel_converter(self) -> Index2Pixel:
		"""
		Getter for attribute **pixel_converter**.

		Returns
		-------
		: Index2Pixel
			The 1D index to 2D pixel converter stored.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.pixel_converter
	
	#--------------------------------------------------------------------------

	def set_pixel_converter(self, pixel_converter: Index2Pixel) -> None:
		"""
		Setter for attribute **pixel_converter**.

		Parameters
		----------
		pixel_converter: Index2Pixel
			The 1D index to 2D pixel converted to use.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.pixel_converter = pixel_converter

	#--------------------------------------------------------------------------

	def get_scale_converter(self) -> PixelPower2ScaleConverter:
		"""
		Getter for attribute **scale_converter**.

		Returns
		-------
		: PixelPower2ScaleConverter
			The pixel scale converted stored.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.scale_converter
	
	#--------------------------------------------------------------------------

	def set_scale_converter(self, scale_converter: PixelPower2ScaleConverter) -> None:
		"""
		Setter for attribute **scale_converter**.

		Parameters
		----------
		scale_converter: PixelPower2ScaleConverter
			The pixel scale converter to use.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.scale_converter = scale_converter

	#--------------------------------------------------------------------------

	def get_index(self) -> int:
		"""
		Getter for attribute **index**.

		Returns
		-------
		: int
			The current index.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.index
	
	#--------------------------------------------------------------------------
	
	def set_index(self, index: int) -> None:
		"""
		Setter for attribute **index**.

		Parameters
		----------
		index: int
			The current index to use.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.index = index

	#--------------------------------------------------------------------------

	def get_added_rows(self) -> int:
		"""
		Getter for attribute **added_rows**.

		Returns
		-------
		: int
			The number of added rows to the stored image.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.added_rows
	
	#--------------------------------------------------------------------------
	
	def set_added_rows(self, added_rows: int) -> None:
		"""
		Setter for attribute **added_rows**.

		Parameters
		----------
		added_rows: int
			The number of added columns to the stored image.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.added_rows = added_rows
	
	#--------------------------------------------------------------------------

	def get_added_columns(self) -> int:
		"""
		Getter for attribute **added_columns**.

		Returns
		-------
		: int
			The number of added columns to the stored image.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.added_columns
	
	#--------------------------------------------------------------------------
	
	def set_added_columns(self, added_columns: int) -> None:
		"""
		Setter for attribute **added_columns**.

		Parameters
		----------
		added_columns: int
			The number of added columns to the stored image.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.added_columns = added_columns

	#--------------------------------------------------------------------------

	def apply_median_filter(self, image: np.ndarray, size: int) -> np.ndarray:
		"""
		Applies a median filter of **size**x**size** to the layer **image**.

		Parameters
		----------
		image: ndarray
			A grayscale image (as Numpy array).
		size: int
			The median filter size.
				
		Returns
		-------
		filtered: ndarray
			The filtered grayscale image.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		filtered = median_filter(image, size=(size,size))
		return filtered

	#--------------------------------------------------------------------------

	def subsample_image(
		self, image: np.ndarray, n: int, method='maximum', 
		use_median_filter=True, median_filter_size=3, 
		invert_image=False
	) -> None:
		"""
		Downsamples the given image n times using the given subsampling method (maximum, mean, minimum).

		Parameters
		----------
		image: ndarray
			The grayscale image to downsample as a Numpy array.
		n: int
			The number of successive downsampling steps to perform.
		method: str
			The downsampling method to use ('maximium', 'minimum' or 'mean').
		use_median_filter: bool
			Whether to apply a median filter to the image before processing it.
		median_filter_size: int
			The size of the median filter, only relevant if `use_median_filter` is True.
		invert_image: bool
			Whether to invert image values (black on white instead as white on black).

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		if invert_image:
			image = 255 - image

		image = self.fix_image_dimensions_for_subdivision(image, n)

		if use_median_filter:
			filtered = self.apply_median_filter(image, 3)
			filtered = normalize_image(filtered)
			self.set_image(filtered)
		else:
			self.set_image(image)
		image = self.get_image()


		data = dict()
		# adding the original image & scale
		scale = 0
		local_data = dict()
		local_data['scale'] = f"1_over_{pow(2,2*scale)}"
		local_data['image'] = image
		data[scale] = local_data

		# subsampling n times by a factor of 2 using the required method
		previous_image = image
		current_image = image
		for i in range(0, n):
			if method == 'maximum':
				current_image = Subsampler.subsample_image_maximum(previous_image)
			elif method == 'minimum':
				current_image = Subsampler.subsample_image_minimum(previous_image)
			elif method == 'mean':
				current_image = Subsampler.subsample_image_mean(previous_image)
			scale = scale + 1

			#current_gradient_image = self.compute_gradient_image(current_image)
			
			local_data = dict()
			local_data['scale'] = f"1_over_{pow(2,2*scale)}"
			local_data['image'] = current_image
			data[scale] = local_data

			previous_image = current_image

		# initialize a pixel and a scale converter
		self.set_data(data)
		pixel_converter = Index2Pixel(current_image.shape[1]) # smallest image
		scale_converter = PixelPower2ScaleConverter()
		self.set_pixel_converter(pixel_converter)
		self.set_scale_converter(scale_converter)
		self.set_index(n)

	#--------------------------------------------------------------------------

	def fix_image_dimensions_for_subdivision(self, image: np.ndarray, n: int) -> np.ndarray:
		"""
		Fixes the image dimensions so that they are divisible by 2^n by adding empty rows
		and columns at the end of it. These added rows/columns have no influence on processing
		as the maximum value will always be taken when performing subdivisions and will be
		removed when the reconstruction step occurs, insuring that the reconstructing image
		matches the original image dimensions. Fills both **added_rows** and **added_columns** attributes.

		Parameters
		----------
		image: ndarray
			The original grayscale image as a Numpy array.
		n: int
			The number of downsampling steps to perform.

		Returns
		-------
		fixed: ndarray
			The grayscale input image but with added empty rows and/or columns.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

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

	def build_base_msct(self, inverted_image=False) -> None:
		"""
		Builds a base MSCT with the lowest scale image.

		Parameters
		----------
		inverted_image: bool
			Whether the image has to be inverted.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		# building a max-tree on the lowest sampled image
		ct = MaxTree()
		image = self.get_image_at_index(self.get_index())
		ct.build_component_tree(image, inverted_image)

		# initializing a MSCT from the max-tree
		msct = MultiScaleComponentTree()
		msct.init_from_component_tree(ct, self.get_pixel_converter(), self.get_index())
		self.set_msct(msct)

	#--------------------------------------------------------------------------

	def compute_mser(self, delta: int) -> None:
		"""
		Computes MSER stabilities for all nodes of the MSCT.

		Parameters
		----------
		delta: int
			The delta parameter for MSER values (gray-level difference).

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		msct = self.get_msct()
		root = msct.get_root()
		msct.compute_mser(node=root, delta=delta)
		#msct.compute_mser_alt(node=root, delta=delta)

	#--------------------------------------------------------------------------

	def compute_mser_percent_height(self, percent_height: float) -> None:
		"""
		Computes MSER stabilities for all nodes of the MSCT using delta as a percentage of the tree height.

		Parameters
		----------
		percent_height: float
			The percentage of the tree height used as delta in [0,1].

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		msct = self.get_msct()
		root = msct.get_root()
		height = msct.compute_max_tree_height()
		delta = max(1, int(round(height * percent_height)))
		msct.compute_mser(node=root, delta=delta)
		#msct.compute_mser_alt(node=root, delta=delta)

	#--------------------------------------------------------------------------

	def augment_msct_mser(self, max_area: int, max_mser: float, invert_image=False) -> None:
		"""
		Augments a set of nodes with one higher scale image using MSER stabilities.
		MSER should be computed prior to running this method !

		Parameters
		----------
		max_area: int
			The maximum area for a flat zone to be considered a valid candidate for enrichment.
		max_mser: float
			The maximum MSER value for a flat zone to be considered a valid candidate for enrichment.
		invert_image: bool
			Whether to invert the image.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		msct = self.get_msct()
		current_scale = self.get_index()
		augment_scale = current_scale - 1
		augment_image = self.get_image_at_index(augment_scale)

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

	def divide_objects(self, max_mser: float, min_area: int) -> list[np.ndarray]:
		"""
		Attempts to identify single objects (nodes) among maximally augmented nodes.

		Parameters
		----------
		max_mser: float
			The maximum MSER value for objects.
		min_area: int
			The minimum surface area for objects.

		Returns
		-------
		objects: list[ndarray]
			The list of clusters being boolean images.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""
		
		objects = identify_single_objects(
			msct=self.get_msct(), 
			scale=self.get_index(), 
			image=self.get_image(), 
			max_mser=max_mser, 
			min_area=min_area
		)

		return objects
	
	#--------------------------------------------------------------------------

	def export_objects_as_images(self, images: list[np.ndarray], output_dir: str, all: bool) -> None:
		"""
		Exports all identified objects as binary images.

		Parameters
		----------
		images: list[ndarray]
			The list of clusters being boolean images.
		output_dir: str
			The path to the output directory.
		all: bool
			Whether to save all individual objects (True) as images or just a combined colour image (False).

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		MSCTObjectDivider.save_objects(
			image=self.get_image(), 
			objects=images, 
			output_dir=output_dir,
			all=all,
			added_rows=self.get_added_rows(),
			added_columns=self.get_added_columns()
		)

	#--------------------------------------------------------------------------

	def export_objects_for_kaggle(self, objects: list) -> list:
		"""
		Exports a list of 2D detected objects as a list of tuples (index_start, length)
		to fit the Kaggle Bowl format.

		Parameters
		----------
		objects: list
		The list of objects, nodes of the MSCT.

		Returns
		-------
		kaggle: list
			The list of tuples (index_start, length) representing `objects`.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		kaggle = []
		msct = self.get_msct()
		# ALGO:
		# build a 2D image
		# crop to the bounding box
		# detect start_index and continue until there is data to obtain length
		# store (start_index, length) in kaggle
		# repeat the two previous steps until reaching the end of the bounding box
		# repreat all above steps for all objects
		for obj in objects:
			fz_pixels = msct.get_local_flat_zone(obj, obj.get_scale())
			fz = msct.build_node_content_image(self.get_image(), fz_pixels)
			width = self.get_image().shape[1]
			start = None
			length = 0
			for row_index in range(0, fz.shape[0]):
				for column_index in range(0, fz.shape[1]):
					if fz[row_index, column_index] != 0:
						if start == None:
							start = row_index * width + column_index
							length = 1
						else:
							length += 1
					else:
						if start != None:
							kaggle.append((start, length))
							start = None
							length = 0
				if length > 0:
					kaggle.append((start, length+1))
					start = None
					length = 0
		return kaggle

	#--------------------------------------------------------------------------

	def save_msct_dot(self, filename: str, simple: bool) -> None:
		"""
		Interface for MultiScaleComponentTree.save_dot() method.

		Parameters
		----------
		filename: str
			The filename where the MSCT is to be saved without the extension.
		simple: bool
			Whether to use the simple saving mode (ID only).

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		msct = self.get_msct()
		msct.save_dot(filename=filename, simple=simple)

	#--------------------------------------------------------------------------

	def save_flat_zones(self, output_dir: str, fz_prefix: str, binary: bool) -> None:
		"""
		Saves all flat zones (one per node) of the MSCT names after their node's ID.

		Parameters
		----------
		output_dir: str
			The output directory for saving.
		fz_prefix: str
			The flat zones prefix name.
		binary: bool
			Whether to save the flat zone as a binary zone or to assign it its original gray level.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		msct = self.get_msct()
		nodes = msct.get_nodes()
		image = self.get_image_at_index(0)
		for node in nodes:
			fz = msct.reconstruct_flat_zone(node, image, binary=binary)
			fz_name = f"{fz_prefix}_{node.get_id()}"
			save_image(image=fz, name=fz_name, output_dir=output_dir)

	#--------------------------------------------------------------------------

	def reconstruct_augmented_nodes(self, output_dir: str, fz_prefix: str, binary: bool) -> None:
		"""
		Reconstructs only augmented nodes.

		Parameters
		----------
		output_dir: str
			The output directory for saving.
		fz_prefix: str
			The flat zones prefix name.
		binary: bool
			Whether to save the flat zone as a binary zone or to assign it its original gray level.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		msct = self.get_msct()
		augmented_nodes = msct.get_augmented_nodes()
		image = self.get_image_at_index(0)

		for node in augmented_nodes:
			aug_fz = msct.reconstruct_flat_zone(node, image, binary=binary)
			aug_fz_name = f"{fz_prefix}_{node.get_id()}"
			save_image(image=aug_fz, name=aug_fz_name, output_dir=output_dir)

	#--------------------------------------------------------------------------

	def reconstruct_all_augmented_nodes(self, output_dir: str, binary: bool) -> None:
		"""
		Reconstructs only augmented nodes.

		Parameters
		----------
		output_dir: str
			The output directory for saving.
		binary: bool
			Whether to save the flat zone as a binary zone or to assign it its original gray level.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		msct = self.get_msct()
		image = self.get_image_at_index(0)
		rec_image = msct.reconstruct_image_augment_nodes_only(image, binary=binary)
		save_image(image=rec_image, name='augmented_nodes', output_dir=output_dir)

	#--------------------------------------------------------------------------

	def save_subsampled_images(self, downsampled_dir: str, downsampled_name: str) -> None:
		"""
		Saves the subsampled images for each scale.

		Parameters
		----------
		downsampled_dir: str
			The output directory for saving downsampled images.
		downsampled_name: str
			The downsampled image prefix name.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		data = self.get_data()
		for index in data.keys():
			image = self.get_image_at_index(index)
			scale = self.get_scale_at_index(index)
			out_name = f"{downsampled_name}_{scale}"
			save_image(image=image, name=out_name, output_dir=downsampled_dir)

	#--------------------------------------------------------------------------

	def reconstruct_image(self, reconstructed_dir: str, reconstructed_name: str, normalize: bool) -> None:
		"""
		Reconstructs the image at the global scale using its current MSCT.

		Parameters
		----------
		reconstructed_dir: str
			The output directory for saving reconstructed images.
		reconstructed_name: str
			The reconstructed image name.
		normalize: bool
			Whether to normalize the image after reconstruction.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		msct = self.get_msct()
		layer_image = self.get_image()
		reconstructed = msct.reconstruct_image(layer_image)
		if normalize:
			reconstructed = self.normalize_image(reconstructed)
		h = reconstructed.shape[0] - self.get_added_rows()
		w = reconstructed.shape[1] - self.get_added_columns()
		image = reconstructed[0:h, 0:w]
		save_image(image=image, name=reconstructed_name, output_dir=reconstructed_dir)

	#--------------------------------------------------------------------------

	def normalize_image(self, image: np.ndarray) -> np.ndarray:
		"""
		Normalizes the signal in range [0,255] for a image.

		Parameters
		----------
		image: ndarray
			A grayscale image as aNumpy array.

		Returns
		-------
		normalized_image: ndarray
			The normalized input image.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		normalized_image = np.zeros(image.shape, dtype=image.dtype)
		amax = np.amax(image)
		if amax == 0:
			normalized_image = image
		else:
			normalized_image = image / amax * 255
		return normalized_image.astype(np.uint8)
