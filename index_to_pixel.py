#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"


#------------------------------------------------------------------------------

class Index2Pixel():
	"""
	This utility class provides methods to convert 1D integer indices to 2D pixels and vice versa.

	Written by Romain PERRIN (<romain.perrin@unistra.fr>).
	"""

	def __init__(self, width: int):
		"""
		Constructor - sets **width** to the user-defined width.

		Parameters
		----------
		width: int
			Width of the image serving as a base for indexing the pixels.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.width = width

	#--------------------------------------------------------------------------

	def get_width(self) -> int:
		"""
		Getter for attribute **width**.

		Returns
		-------
		: int
			Width of the image serving as a base for indexing the pixels.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.width
	
	#--------------------------------------------------------------------------

	def set_width(self, width: int) -> None:
		"""
		Setter for attribute **width**.

		Parameters
		----------
		width: int
			Width of the image serving as a base for indexing the pixels.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.width = width

	#--------------------------------------------------------------------------

	def convert_pixel_to_index(self, pixel: tuple[int, int]) -> int:
		"""
		Converts a pixel represented by a tuple to an index

		Parameters
		----------
		pixel: tuple[int, int]
			2D integer pixel designated by its (row, column) coordinate. 

		Returns
		-------
		index: int
			1D index of the input 2D **pixel** in the current referential.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		row, col = pixel
		index = row * self.get_width() + col
		return index
	
	#--------------------------------------------------------------------------

	def convert_pixels_to_indices(self, pixels: set[tuple[int, int]]) -> set[int]:
		"""
		Converts a set of 2D pixels to a set of 1D pixel indices in the current referential.

		Parameters
		----------
		pixels: set[tuple[int, int]]
			Set of 2D pixels represented by their (row, column) coordinates.

		Returns
		-------
		indices: set[int]
			Set of 1D pixel indices representing the 2D indices in the current referential.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		indices = set()
		for pixel in pixels:
			indices.add(self.convert_pixel_to_index(pixel))
		return indices

	#--------------------------------------------------------------------------

	def convert_index_to_pixel(self, index: int) -> tuple[int, int]:
		"""
		Converts a pixel represented by a 1D index to a tuple of 'row, column) in the current referential.

		Parameters
		----------
		index: int
			1D index of the pixel in the current referential.

		Returns
		-------
		tuple[int, int]
			2D pixel represented by its (row, column) coordinate from the current referential.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		row = index // self.get_width()
		column = index % self.get_width()
		return (row, column)

	#--------------------------------------------------------------------------

	def convert_indices_to_pixels(self, indices: set[int]) -> set[tuple[int, int]]:
		"""
		Converts all 1D pixel indices in the current referential to a set of 2D (row, column) pixels.

		Parameters
		----------
		indices: set[int]
			Set of 1D pixel indices in the current referential.

		Returns
		-------
		all_pixels: set[tuple[int, int]]
			Set of 2D pixels represented by their (row, column) coordiantes from the current referential.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		all_pixels = set()
		for index in indices:
			pixel = self.convert_index_to_pixel(index)
			all_pixels.add(pixel)
		return all_pixels