#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import numpy as np
import pydot

#------------------------------------------------------------------------------

class ComponentTree():
	"""
	Represents a component-tree, with a root node and a list of nodes (ComponentTreeNode).

	Attributes
	----------
	root : int
		The index of the root node.
	nodes : list[ComponentTreeNode]
		A list of nodes (*ComponentTreeNode*).
	invert : bool
		Whether the image has to be inverted (default is bright objects over a dark background).

	Methods
	-------
	print_tree(node):
		Prints the tree structure in the console starting at the given node.
	save_dot(filename):
		Writes the component-tree in the DOT language into a dot file.
	build_component_tree(image, invert):
		Builds the component-tree of the given image using Najman's algorithm.
	build_component_tree_from_partial_image(image, mask, invert):
		Builds the component-tree of a partial image using Najman's algorithm.
	reconstruct_image(image):
		Reconstructs an image using the its component-tree.

	Written by Romain PERRIN (<romain.perrin@unistra.fr>).
	"""

	def __init__(self):
		"""
		Constructor - initializes the root to 0, creates an empty list of nodes and sets invert to *False*.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.root = 0 # index of the root
		self.nodes = list()  # list of nodes
		self.invert = False

	#--------------------------------------------------------------------------

	def get_root(self) -> int:
		"""
		Getter for attribute **root**.

		Returns
		-------
		: int
			The ID of the component tree's root node.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.root
	
	#--------------------------------------------------------------------------

	def set_root(self, root: int):
		"""
		Setter for attribute **root**.

		Parameters
		----------
		root : int
			The index of the component tree's root node.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.root = root

	#--------------------------------------------------------------------------

	def get_nodes(self) -> list:
		"""
		Getter for attribute **nodes**.

		Returns
		-------
		: list[ComponentTreeNode]
			The list of nodes (including the root), objects are instances of ComponentTreeNode.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.nodes
	
	#--------------------------------------------------------------------------

	def set_nodes(self, nodes: list) -> None:
		"""
		Setter for attribute **nodes**.

		Parameters
		----------
		nodes : list[ComponentTreeNode]
			The list of nodes as a list of ComponentTreeNode objects.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.nodes = nodes

	#--------------------------------------------------------------------------

	def get_invert(self) -> bool:
		"""
		Getter for attribute **invert**.

		Returns
		-------
		: bool
			*True* if the component-tree is built for bright objects over a dark background, *False* otherwise.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.invert
	
	#--------------------------------------------------------------------------

	def set_invert(self, invert: bool) -> None:
		"""
		Setter for attribute **invert**.

		Parameters
		----------
		invert: bool
			*True* is the component-tree is built for bright objects over a dark background, *False* otherwise.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.invert = invert

	#--------------------------------------------------------------------------

	def print_tree(self, node: int) -> None:
		"""
		Displays the global structure of the component-tree starting at a given node.

		Parameters
		----------
		node: int
			The index of the root node (can be any node in the component-tree).

		Returns
		-------
		Prints on the standard output.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		print(f"# Node ({node}) :")
		print(f"    - Level   -> {self.get_nodes()[node].get_level()}")
		print(f"    - Highest   -> {self.get_nodes()[node].get_highest()}")
		print(f"    - Area   -> {self.get_nodes()[node].get_area()}")
		print(f"    - Father   -> {self.get_nodes()[node].get_father()}")
		print(f"    - Children   -> [")
		for child in self.get_nodes()[node].get_children():
			print(f"{child}, ")
		print(f"]")

		for child in self.get_nodes()[node].get_children():
			self.print_tree(child)

	#--------------------------------------------------------------------------

	def save_dot(self, filename: str) -> None:
		"""
		Saves the current component-tree to a dot file using the DOT language and pyDot/GraphViz.

		Parameters
		----------
		filename: str
			The path to the output file without the extension.

		Returns
		-------
		None

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		graph = pydot.Dot('component_tree', graph_type='graph', bgcolor='white')

		# add all nodes from root to leaves
		for node in self.get_nodes():
			node = pydot.Node(name=f"node_{node.get_id()}")
			graph.add_node(node)

		# connect nodes
		process = set()
		process.add(self.get_nodes()[self.get_root()])

		while len(process) > 0:
			cur_node = process.pop()
			for child_index in cur_node.get_children():
				child_node = self.get_nodes()[child_index]
				process.add(child_node)
				edge = pydot.Edge(f"node_{cur_node.get_id()}", f"node_{child_node.get_id()}")
				graph.add_edge(edge)

		#graph.write_dot(f"{filename}.dot")
		graph.write_png(f"{filename}.png")

	#--------------------------------------------------------------------------

	def build_component_tree(self, image: np.ndarray, invert_image: bool) -> None:
		"""
		Builds a component-tree using an implementation based on Najman's algorithm published in :
		L.Najman, M.Croupie, "Building the component-tree in quasi-linear time", Vol. 15, Num. 11, p. 3531-3539, 2006.

		Parameters
		----------
		image: ndarray
			The grayscale image on which to build the component-tree (as a Numpy array).
		invert_image: bool
			Whether the input image should be inverted (use *True* for bright objects on a dark background, *False* for the opposite).

		Returns
		-------
		None

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		# placeholder, implementation in derived classes
		pass

	#--------------------------------------------------------------------------

	def build_component_tree_from_partial_image(self, image: np.ndarray, mask:np.ndarray, invert_image: bool) -> None:
		"""
		Builds a component-tree on a given set of pixels using an implementation based on Najman's algorithm published in :
		L.Najman, M.Croupie, "Building the component-tree in quasi-linear time", Vol. 15, Num. 11, p. 3531-3539, 2006.
		Same principle as *build_component_tree()* but on a given subset of pixels of the input image.

		Parameters
		----------
		image: ndarray:
			The grayscale image on which to build the component-tree (as aNumpy array).
		mask: ndarray
			A mask image of the same size as **image** indicating which pixels should be processed (Numpy array).
		invert_image: bool
			Whether the input image should be inverted (use *True* for bright objects on a dark background, *False* for the opposite).

		Returns
		-------
		None

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		# placeholder, implementation in derived classes
		pass

	#--------------------------------------------------------------------------

	def reconstruct_image(self, image: np.ndarray) -> np.ndarray:
		"""
		Reconstructs the image using its component-tree.

		Parameters
		----------
		image: ndarray
			The Numpy array of the original image (used to set the resulting image' shape and type)

		Returns
		-------
		reconstructed: ndarray
			The grayscale reconstructed image from the stored component-tree as a Numpy array (same shape and type as **image**).

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		reconstructed = np.zeros(image.shape, dtype=np.uint8)
		reconstructed = reconstructed.flatten()

		to_process = []
		root_node = self.nodes()[self.root()]
		for pixel in root_node.get_pixels():
			reconstructed[pixel] = root_node.get_level()
		for child_index in root_node.get_children():
			to_process.append(child_index)

		while len(to_process) > 0:
			node_index = to_process.pop(0)
			node = self.nodes()[node_index]
			for pixel in node.get_pixels():
				reconstructed[pixel] = node.get_level()
			for child_index in node.get_children():
				to_process.append(child_index)

		reconstructed = reconstructed.reshape(image.shape)

		if self.get_invert():
			reconstructed = 255 - reconstructed

		return reconstructed
