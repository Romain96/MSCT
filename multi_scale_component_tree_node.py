#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

from typing_extensions import Self

#------------------------------------------------------------------------------

class MultiScaleComponentTreeNode():
	"""
	Represents a node of a multiscale component-tree (MSCT).

	Written by Romain PERRIN (<romain.perrin@unistra.fr>).
	"""

	_id_generator = 0
		
	def __init__(self):
		"""
		Constructor - generates a unique ID, initializes all integers to 0 and sets to empty.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.id = MultiScaleComponentTreeNode._id_generator
		MultiScaleComponentTreeNode._id_generator += 1
		self.active = True

		self.level = None
		self.father = MultiScaleComponentTreeNode
		self.children = set()
		self.pixels = dict()
		# node area and subtree area at the global scale
		self.area = 0
		self.subarea = 0
		# MSER stability value
		self.mser = 0
		# multi channel histogram
		self.histogram = []

	#--------------------------------------------------------------------------

	def get_id(self) -> int:
		"""
		Getter for attribute **id**.

		Returns
		-------
		: int
			The node's unique ID.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""
		return self.id
	
	#--------------------------------------------------------------------------

	def set_id(self, id: int) -> None:
		"""
		Setter for attribute **id**."
		
		Parameters
		----------
		id: int
			The node's unique ID.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.id = id

	#--------------------------------------------------------------------------

	def get_scales(self) -> list[int]:
		"""
		Returns a list of all unique scales associated with the node's pixels.

		: list[int]
			The list of all unique scales of the node's pixels.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""
		
		return self.pixels.keys()

	#--------------------------------------------------------------------------

	def get_level(self) -> int:
		"""
		Getter for attribute **level**.

		Returns
		-------
		: int
			The gray level associated with the node.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.level
	
	#--------------------------------------------------------------------------

	def set_level(self, level: int) -> None:
		"""
		Setter for attribute **level**.

		Parameters
		----------
		level: int
			The gray level associated with the node.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.level = level

	#--------------------------------------------------------------------------

	def get_father(self) -> Self:
		"""
		Getter for attribute **father**.

		Returns
		-------
		: MultiScaleComponentTreeNode
			The node's father.
		
		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.father
	
	#--------------------------------------------------------------------------

	def set_father(self, father: Self) -> None:
		"""
		Setter for attribute **father**.

		Parameters
		----------
		father: MultiScaleComponentTreeNode
			The node's father.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.father = father

	#--------------------------------------------------------------------------

	def get_children(self) -> set[Self]:
		"""
		Getter for attribute **children**.

		Returns
		-------
		children: set[MultiScaleComponentTreeNode]
			The set of all the node's (direct) children.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.children
	
	#--------------------------------------------------------------------------

	def set_children(self, children: set[Self]) -> None:
		"""
		Setter for attribute **children**.

		Parameters
		----------
		children: set[MultiScaleComponentTreeNode]
			The set of all the node's (direct) children.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.children = children

	#--------------------------------------------------------------------------

	def get_pixels(self) -> dict:
		"""
		Getter for attribute **pixels**.

		Returns
		-------
		: dict
			A dictionary with sets of pixels (values) per image scale (keys).

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.pixels
	
	#--------------------------------------------------------------------------

	def set_pixels(self, pixels: dict) -> None:
		"""
		Setter for attribute **pixels**.

		Parameters
		----------
		pixels: dict
			A dictionary with sets of pixels (value) per image scale (keys).

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.pixels = pixels

	#--------------------------------------------------------------------------

	def get_pixels_at_scale(self, scale: int) -> set:
		"""
		Returns the set of pixels contained in the node at the given scale.

		Parameters
		----------
		scale: int
			The image scale.
		
		Returns
		-------
		: set
			The set of all pixels contained in the node and that are in the given **scale**.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		if scale in self.pixels.keys():
			return self.pixels[scale]
		raise Exception(f"The given scale {scale} does not exist in the current node {self.id} (scales = {self.pixels.keys()})")
	
	#--------------------------------------------------------------------------

	def set_pixels_at_scale(self, pixels: set, scale: int) -> None:
		"""
		Replaces the set of pixels at the given scale by the given set if it exists or creates it otherwise.

		Parameters
		----------
		pixels: set
			Set of all pixels at **scale**.
		scale: int
			The scale of **pixels**.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.pixels[scale] = pixels

	#--------------------------------------------------------------------------

	def merge_pixels_at_scale(self, pixels: set, scale: int) -> None:
		"""
		Merges the given set of pixels at the given scale with the existing one.

		Parameters
		----------
		pixels: set
			The set of new pixels at **scale**.
		scale: int
			The scale of **pixels**.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		if scale in self.get_scales():
			merged_pixels = self.get_pixels_at_scale(scale)
			merged_pixels.update(pixels)
			self.set_pixels_at_scale(merged_pixels, scale)
		else:
			self.set_pixels_at_scale(pixels, scale)

	#--------------------------------------------------------------------------

	def get_area(self) -> int:
		"""
		Returns the node's pixel area at the global scale.

		: int
			The node's pixel area at the global scale.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.area
	
	#--------------------------------------------------------------------------

	def set_area(self, area: int) -> None:
		"""
		Sets the node's pixel area at the global scale.

		Parameters
		----------
		area: int
			The node's pixel area at the global scale.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.area = area

	#--------------------------------------------------------------------------

	def compute_area(self) -> None:
		"""
		Computes the node's pixel area at the global scale using its pixels dictionnary.
		Sets the **area** attribute of the node.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		area = 0
		for key in self.pixels.keys():
			area += pow(4, key) * len(self.get_pixels_at_scale(scale=key))
		self.set_area(area)

	#--------------------------------------------------------------------------

	def get_subarea(self) -> int:
		"""
		Returns the node's pixel area of its subtree excluding the current node at the global scale.

		Returns
		-------
		: int
			The node's subtree pixel area at the global scale excluding itself.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.subarea
	
	#--------------------------------------------------------------------------

	def set_subarea(self, subarea: int) -> None:
		"""
		Sets the node's pixel area of its subtree excluding the current node at the global scale.

		Parameters
		----------
		subarea: int
			The node's subtree pixel area at the global scale excluding itself.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.subarea = subarea

	#--------------------------------------------------------------------------

	def get_mser(self) -> float:
		"""
		Getter for attribute **mser** (Maximally Stable Extremal Regions).

		Returns
		-------
		: float
			The node's MSER stability value.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.mser
	
	#--------------------------------------------------------------------------

	def set_mser(self, mser: float) -> None:
		"""
		Setter for attribute **mser** (Maximally Stable Extremal Regions).

		Parameters
		----------
		mser: float
			THe node's MSER stability value.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.mser = mser

	#--------------------------------------------------------------------------

	def get_active(self) -> bool:
		"""
		Getter for attribute **active**.
		Used to indicate whether a node is active after a filtering step.

		Returns
		-------
		: bool
			*True* if the node is active, *False* otherwise.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.active
	
	#--------------------------------------------------------------------------

	def set_active(self, active: bool) -> None:
		"""
		Setter for attribute **active**.
		Used to indicate whether a node is active after a filtering step.

		Parameters
		----------
		active: bool
			*True* is the node is active, *False* otherwise.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.active = active

	#--------------------------------------------------------------------------

	def get_histogram(self) -> list[int]:
		"""
		Getter for attribute **histogram**.

		Returns
		-------
		: list[int]
			The node's histogram.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.histogram
	
	#--------------------------------------------------------------------------
	
	def set_histogram(self, histogram: list[int]) -> None:
		"""
		Setter for attribute **histogram**.

		Parameters
		----------
		histogram: list[int]
			The node's histogram.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.histogram = histogram

	#--------------------------------------------------------------------------

	def add_child(self, child: Self) -> None:
		"""
		Adds a child to the node' set of children.

		Parameters
		----------
		child: MultiScaleComponentTreeNode
			The child node to add.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.children.add(child)

	#--------------------------------------------------------------------------

	def remove_child(self, child: Self) -> None:
		"""
		Removes a child to the node' set of children.

		Parameters
		----------
		child: MultiScaleComponentTreeNode
			The child node to remove.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.children.remove(child)

	#--------------------------------------------------------------------------

	def get_total_area(self) -> int:
		"""
		Returns the node's pixel area plus its subtree pixel area at the global scale.

		Returns
		-------
		: int
			The node's total pixel area (own area + subtree area) at the global scale.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.get_area() + self.get_subarea()

	#--------------------------------------------------------------------------

	def is_leaf(self) -> bool:
		"""
		Returns *True* if the node is a leaf (doesn't have any child), *False* otherwise.

		Returns
		-------
		: bool
			*True* if the node is a leaf, *False* otherwise.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		if len(self.get_children()) == 0:
			return True
		else:
			return False
		
	#--------------------------------------------------------------------------

	def is_descendent_of(self, other: Self) -> bool:
		"""
		Returns *True* if the current node is a descendent of the node given in parameter, *False* otherwise.

		Parameters
		----------
		other: MultiScaleComponentTreeNode
			The other node for comparison.

		Returns
		-------
		: bool
			*True* if the current node is a descendant of **other**, *False* otherwise.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		if self.get_father() == self:
			return False
		elif self.get_father() == other:
			return True
		else:
			return self.get_father().is_descendent_of(other)
		
	#--------------------------------------------------------------------------

	def is_ascendent_of(self, other: Self) -> bool:
		"""
		Returns *True* if the current node is a ascendent of the node given in parameter, *False* otherwise.

		Parameters
		----------
		other: MultiScaleComponentTreeNode
			The other node for comparison.

		Returns
		-------
		: bool
			*True* if the current node is an ascendant of **other**, *False* otherwise.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		if other.get_father() == other.get_father():
			return False
		elif other.get_father() == self:
			return True
		else:
			return self.is_ascendent_of(other.get_father())
	
