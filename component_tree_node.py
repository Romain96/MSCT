#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

#------------------------------------------------------------------------------

class ComponentTreeNode():
	"""
	A class representing a node of a component-tree.

	Attributes
	----------
	id : int
		unique ID of the node.
	index : int
		index of the node.
	level : int
		gray level of the node.
	highest : int
		highest gray level of the node.
	area : int
		pixel area of the node.
	father : int
		index of the father node.
	children : set
		set of indices of children nodes.
	pixels : set
		set of indices of pixels (ComponentTreePoint).

	Written by Romain PERRIN (<romain.perrin@unistra.fr>).
	"""

	_id_generator = 0

	def __init__(self):
		"""
		Constructor - initializes the unique ID, initializes all remaining integers to 0 and sets to empty.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.id = ComponentTreeNode._id_generator
		ComponentTreeNode._id_generator += 1

		self.index = 0
		self.level = 0
		self.highest = 0
		self.area = 0
		self.father = 0
		self.subarea = 0
		self.children = set()
		self.pixels = set()

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
		Setter for attribute **id**.

		Parameters
		----------
		id: int
			The node's unique ID.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.id = id

	#--------------------------------------------------------------------------

	def get_index(self) -> int:
		"""
		Getter for attribute **index**.

		Returns
		-------
		: int
			The node's index.

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
			The node's index.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.index = index

	#--------------------------------------------------------------------------

	def get_level(self) -> int:
		"""
		Getter for attribute **level**.

		Returns
		-------
		: int
			The node's level.

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
			The node's level.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.level = level

	#--------------------------------------------------------------------------

	def get_highest(self) -> int:
		"""
		Getter for attribute **highest**.

		Returns
		-------
		: int
			The node's highest gray level.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.highest
	
	#--------------------------------------------------------------------------

	def set_highest(self, highest: int) -> None:
		"""
		Setter for attribute **highest**.

		Parameters
		----------
		highest: int
			The node's highest gray level.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.highest = highest

	#--------------------------------------------------------------------------
	
	def get_area(self) -> int:
		"""
		Getter for attribute **area**.

		Returns
		-------
		: int
			The node's own surface area (number of pixels belonging to the node only).

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.area
	
	#--------------------------------------------------------------------------

	def set_area(self, area: int) -> None:
		"""
		Setter for attribute **area**.

		Parameters
		----------
		area: int
			The node's own surface area (number of pixels belonging to the node only).

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""
		self.area = area

	#--------------------------------------------------------------------------

	def get_subarea(self) -> int:
		"""
		Getter for attribute **subarea**.

		Returns
		-------
		: int
			The node's subarea (number of pixels in the subtree minus the node itself).

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.subarea
	
	#--------------------------------------------------------------------------
	
	def set_subarea(self, subarea: int) -> None:
		"""
		Setter for attribute **subarea**.

		Parameters
		----------
		subarea: int
			The node's subarea (number of pixels in the subtree minus the node itself).

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.subarea = subarea

	#--------------------------------------------------------------------------

	def get_father(self) -> int:
		"""
		Getter for attribute **father**.

		Returns
		-------
		: int
			The node's father index.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.father

	#--------------------------------------------------------------------------

	def set_father(self, father: int) -> None:
		"""
		Setter for attribute **father**.

		Parameters
		----------
		father: int
			The node's father index.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.father = father

	#--------------------------------------------------------------------------

	def get_children(self) -> set:
		"""
		Getter for attribute **children**.

		Returns
		-------
		: set[ComponentTreeNode]
			The set of direct children nodes (ComponentTreeNode)

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.children

	#--------------------------------------------------------------------------

	def set_children(self, children: set) -> None:
		"""
		Setter for attribute **children**.

		Parameters
		----------
		children: set[ComponentTreeNode]
			The set of directe children nodes (ComponentTreeNode).

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.children = children

	#--------------------------------------------------------------------------

	def get_pixels(self) -> set:
		"""
		Getter for attribute **pixels**.

		Returns
		-------
		: set[ComponentTreePoint]
			The set of all pixels belonging to the node.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.pixels
	
	#--------------------------------------------------------------------------

	def set_pixels(self, pixels: set) -> None:
		"""
		Setter for attribute **pixels**.

		Parameters
		----------
		pixels: set[ComponentTreePoint]
			The set of all pixels belonging to the node.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.pixels = pixels

	#--------------------------------------------------------------------------

	def add_pixel(self, pixel: int) -> None:
		"""
		Adds a pixel to the node' set of pixels.

		Parameters
		----------
		pixel: int
			A pixel represented by its index.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""
		
		self.pixels.add(pixel)
