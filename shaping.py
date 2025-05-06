#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import pydot
from multi_scale_component_tree import MultiScaleComponentTree
from multi_scale_component_tree_node import MultiScaleComponentTreeNode
from mser_tree import MSERNode, MSERTree

#------------------------------------------------------------------------------

class ShapingTreePoint:
	"""
	This class implements a point of a shaping on the MSCT

	Attributes
	----------

	index: int
		The point's unique index.
	value: int 
		The point's value (gray-level).
	
	Methods
	-------

	Written by Romain PERRIN (<romain.perrin@unistra.fr>).
	"""

	def __init__(self):
		"""
		Initializes the shaping container.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""
		
		self.index = None
		self.value = None

	#--------------------------------------------------------------------------

	def get_index(self) -> int:
		"""
		Getter for attribute **index**.

		Returns
		-------
		: int
			The point's index.

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
			The point's index.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.index = index

	#--------------------------------------------------------------------------

	def get_value(self) -> int:
		"""
		Getter for attribute **value**.

		Returns
		-------
		: int
			The point's gray-level value.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.value
	
	#--------------------------------------------------------------------------

	def set_value(self, value: int) -> None:
		"""
		Setter for attribute **value**.

		Parameters
		----------
		value: int
			The point's gray-level value.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.value = value

#------------------------------------------------------------------------------

class ShapingTreeNode:
	"""
	This class represents a node of a shaping on a MSCT i.e. a max-tree node built on a MSCT.

	Attributes
	----------
	index: int
		Index the node.
	mser: floar
		MSER stability value of the node.
	father: int
		The index of the node's father.
	children: set
		The set of the node's children nodes.
	nodes: set
		The set of all MSCT nodes contained in this node.
	
	Methods
	-------
	add_node(MultiScaleComponentTreeNode)

	Written by Romain PERRIN (<romain.perrin@unistra.fr>).
	"""

	_idgen = 0

	def __init__(self):
		"""
		Initializes an empty node.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.id = ShapingTreeNode._idgen
		ShapingTreeNode._idgen += 1

		self.index = 0
		self.mser = 0
		self.father = 0
		self.children = set()
		self.nodes = set()

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

	def get_mser(self) -> float:
		"""
		Getter for attribute **mser**.

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
		Setter for attribute **mser**.

		Parameters
		----------
		mser: float
			The node's MSER stability value.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.mser = mser

	#--------------------------------------------------------------------------

	def get_father(self) -> int:
		"""
		Getter for attribute **father**.

		Returns
		-------
		: int
			The index of the node's father.

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
			The index of the node's father.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.father = father

	#--------------------------------------------------------------------------

	def get_children(self) -> set:
		"""
		Getter for attribute **children**.

		Returns
		-------
		: set
			The set of the node's children nodes.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.children
	
	#--------------------------------------------------------------------------

	def set_children(self, children: set) -> None:
		"""
		Setter for attribute **children**.

		Parameters
		----------
		children: set
			The set of the node's children nodes.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.children = children

	#--------------------------------------------------------------------------

	def get_nodes(self) -> set:
		"""
		Getter for attribute **nodes**.

		Returns
		-------
		: set
			The set of the node's MSCT nodes.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.nodes
	
	#--------------------------------------------------------------------------

	def set_nodes(self, nodes: set) -> None:
		"""
		Setter for attribute **nodes**.

		Parameters
		----------
		nodes: set
			The set of the node's MSCT nodes.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""
		
		self.nodes = nodes

	#--------------------------------------------------------------------------

	def add_node(self, node: MultiScaleComponentTreeNode) -> None:
		"""
		Adds a MSCT node to the set of nodes.

		Parameters
		----------
		node: MultiScaleComponentTreeNode
			A MSCT node to add.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.nodes.add(node)

#------------------------------------------------------------------------------

class ShapingTreeSet:
	"""
	This class represents a set in the shaping i.e. a max-tree built on a MSCT.

	Attributes
	----------
	parent: int
		The index of the set's parent.
	rank: int
		The rank of the set.

	Methods
	-------

	Written by Romain PERRIN (<romain.perrin@unistra.fr>).
	"""

	def __init__(self):
		"""
		Initializes an empty set.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.parent = None
		self.rank = None

	#--------------------------------------------------------------------------

	def get_parent(self) -> int:
		"""
		Getter for attribute **parent**.

		Returns
		-------
		: int
			The set's parent.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.parent
	
	#--------------------------------------------------------------------------

	def set_parent(self, parent: int) -> None:
		"""
		Setter for attribute **parent**.

		Parameters
		----------
		parent: int
			The set's parent.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.parent = parent

	#--------------------------------------------------------------------------

	def get_rank(self) -> int:
		"""
		Getter for attribute **rank**.

		Returns
		-------
		: int
			The set's rank.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.rank
	
	#--------------------------------------------------------------------------

	def set_rank(self, rank: int) -> None:
		"""
		Setter for attribute **rank**.

		Parameters
		----------
		rank: int
			The set's rank.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.rank = rank

#------------------------------------------------------------------------------

def make_node(point: ShapingTreePoint, node: MultiScaleComponentTreeNode) -> ShapingTreeNode:
	"""
	Tarjan's union-find make node.

	Parameters
	----------
	point: ShapingTreePoint
		A point.
	node: MultiScaleComponentTreeNode
		A MSCT node.

	Returns
	-------
	n: ShapingTreeNode
		The resulting shaping tree node.

	Written by Romain PERRIN (<romain.perrin@unistra.fr>).
	"""

	n = ShapingTreeNode()
	n.set_index(point.get_index())
	n.set_mser(point.get_value())
	n.set_father(point.get_index())
	n.set_children(set())
	n.add_node(node)
	return n

#------------------------------------------------------------------------------

def make_set(x: int) -> ShapingTreeSet:
	"""
	Add the set {x} to the collection Q, provided that the element x does not already belongs to a set in Q.

	Parameters
	----------
	x: int
		The set {x}.

	Returns
	-------
	s: ShapingTreeSet
		The set {x}.

	Written by Romain PERRIN (<romain.perrin@unistra.fr>).
	"""

	s = ShapingTreeSet()
	s.set_parent(x)
	s.set_rank(0)
	return s

#------------------------------------------------------------------------------

def find(q: list, x: int) -> int:
	"""
	Return the canonical element of the set in Q which contains x.

	Parameters
	----------
	q: list
		The collection Q.
	x: int
		The canonical element x.

	Returns
	-------
	find: int
		The canonical element of Q containing x.

	Written by Romain PERRIN (<romain.perrin@unistra.fr>).
	"""

	if (q[x].get_parent() != x):
		q[x].set_parent(find(q, q[x].get_parent()))

	return q[x].get_parent()

#------------------------------------------------------------------------------

def link(q: list, x: int, y: int) -> int:
	"""
	Let X and Y be the two sets in Q whose canonical elements are x and y respectively (x and y must be different). 
	Both sets are removed from Q, their union Z = X ∪ Y is added to Q and a canonical element for Z is selected and returned.

	Parameters
	----------
	q: list
		The collection Q.
	x: int
		The canonical element x.
	y: int
		The canonical element y.

	Returns
	-------
	y: int
		The canonical element of Z = X ∪ Y.

	Written by Romain PERRIN (<romain.perrin@unistra.fr>).
	"""

	if (q[x].get_rank() > q[y].get_rank()):
		x, y = y, x
		
	if (q[x].get_rank() == q[y].get_rank()):
		q[y].set_rank(q[y].get_rank() + 1)

	q[x].set_parent(y)
	return y

#------------------------------------------------------------------------------


def merge_nodes(nodes: list[ShapingTreeNode], q: ShapingTreeSet, n1: int, n2: int) -> int:
	"""
	Merge two nodes (attributes, etc.) together and return the index of the resulting node.

	Parameters
	----------
	nodes: list[ShapingTreeNode]
		The list of nodes.
	q: ComponentTreeSet
		The collection Q.
	n1: int
		Index of the first node.
	n2: int
		Index of the second node.

	Returns
	-------
	tmp_n1: int
		The index of the resulting node.

	Written by Romain PERRIN (<romain.perrin@unistra.fr>).
	"""

	tmp_n1 = link(q, n1, n2)
	if tmp_n1 == n2:
		tmp_n2 = n1
	else:
		tmp_n2 = n2

	# update attributes
	nodes[tmp_n1].get_nodes().update(nodes[tmp_n2].get_nodes())
	nodes[tmp_n2].get_nodes().clear()

	# add the list of children of the node that is not kept to the list of children of the node that is kept
	for child in nodes[tmp_n2].get_children():
		nodes[tmp_n1].get_children().add(child)
		nodes[child].set_father(tmp_n1)

	nodes[tmp_n2].set_children([])
	return tmp_n1

#------------------------------------------------------------------------------

class MSCTShapingTree:
	"""
	This class provides a storage for a MSCT shaping tree, that is a max-tree build
	on a MSCT to implement the principle of *shaping*.

	Attributes
	----------
	id: int
		The tree's ID.
	root: ShapingTreeNode
		The tree's root node.
	nodes: set[ShapingTreeNode]
		The set of all the tree's nodes.
	
	Methods
	-------
	add_node(ShapingTreeNode)
		Adds a new node to the tree.
	remove_node(ShapingTreeNode)
		Removes an existing node from the tree.

	Written by Romain PERRIN (<romain.perrin@unistra.fr>).
	"""

	_idgen = 0

	def __init__(self):
		"""
		Initializes an empty tree.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.id = MSCTShapingTree._idgen
		MSCTShapingTree._idgen += 1
		self.root = None
		self.nodes = set()

	#--------------------------------------------------------------------------

	def get_root(self) -> ShapingTreeNode:
		"""
		Getter for attribute **root**.

		Returns
		-------
		: ShapingTreeNode
			The root node of the tree.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.root
	
	#--------------------------------------------------------------------------
	
	def set_root(self, root: ShapingTreeNode) -> None:
		"""
		Setter for attribute **root**.

		Parameters
		----------
		root: ShapingTreeNode
			The root node of the tree.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.root = root

	#--------------------------------------------------------------------------

	def get_nodes(self) -> set[ShapingTreeNode]:
		"""
		Getter for attribute **nodes**.

		Returns
		-------
		: set[ShapingTreeNode]
			The set of all the tree's nodes.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.nodes
	
	#--------------------------------------------------------------------------
	
	def set_nodes(self, nodes: set[ShapingTreeNode]) -> None:
		"""
		Setter for attribute **nodes**.

		Parameters
		----------
		nodes: set[ShapingTreeNode]
			The set of all the tree's nodes.
		
		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.nodes = nodes

	#--------------------------------------------------------------------------

	def add_node(self, node: ShapingTreeNode) -> None:
		"""
		Adds a node to the tree.

		Parameters
		----------
		node: ShapingTreeNode
			A node to add.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.get_nodes().add(node)

	#--------------------------------------------------------------------------

	def remove_node(self, node: ShapingTreeNode) -> None:
		"""
		Removes a node from the tree.

		Parameters
		----------
		node: ShapingTreeNode
			A node to remove.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.get_nodes().remove(node)

	#--------------------------------------------------------------------------

	def filter_return_leq(self, mser: float) -> set[MSERNode]:
		"""
		Returns only nodes of the tree whose MSER value is lesser or equal to **mser**.
		MSCT nodes contained are returned instead of MSCTShapingTreeNode.

		Parameters
		----------
		mser: float
			Maximum MSER stability value of nodes to keep.

		Returns
		-------
		msct_nodes: set[MSERNode]
			Set of simplified MSERNodes.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		to_process = []
		to_process.append(self.get_root())
		msct_nodes = set()
		while len(to_process) > 0:
			node = to_process.pop(0)
			if node.get_mser() <= mser:
				for n in node.get_nodes():
					msct_nodes.add(n)
			for child in node.get_children():
				to_process.append(child)
		return msct_nodes

	#----------------------------------------

	def save_dot(self, filename: str) -> None:
		"""
		Saves the current tree to a dot file using the DOT language and pyDot/GraphViz.

		Parameters
		----------
		filename: str
			The path to the output file without the extension.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		graph = pydot.Dot('MSCTShapingTree', graph_type='graph', bgcolor='white')

		# add all nodes from root to leaves
		for node in self.get_nodes():
			infos = [i.get_id() for i in node.get_nodes()]
			node = pydot.Node(name=f"node_{node.get_id()}", label=f"ID {node.get_id()}\n{node.get_mser()}\n({infos})")
			graph.add_node(node)

		# connect nodes
		process = []
		process.append(self.get_root())

		while len(process) > 0:
			cur_node = process.pop()
			for child in cur_node.get_children():
				process.append(child)
				edge = pydot.Edge(f"node_{cur_node.get_id()}", f"node_{child.get_id()}")
				graph.add_edge(edge)

		#graph.write_dot(f"{filename}.dot")
		graph.write_png(f"{filename}.png")

#------------------------------------------------------------------------------


class MSCTShapingTreeNode:
	"""
	A node of the shaping tree.

	Attributes
	----------
	father: int
		The node's father.
	children: set
		The node's set of children nodes.
	nodes: set
		The set of MSCT nodes.
	mser: float
		The MSER stability value.

	Methods
	-------
	add_node()
		Adds a node to the tree.
	add_child()
		Adds a child to a node.

	Written by Romain PERRIN (<romain.perrin@unistra.fr>).
	"""

	_idgen = 0

	def __init__(self):
		"""
		Initializes an empty node.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.id = MSCTShapingTreeNode._idgen
		MSCTShapingTreeNode._idgen += 1

		self.father = None
		self.children = set()
		self.nodes = set()
		self.mser = 0.0

	#--------------------------------------------------------------------------

	def get_id(self) -> int:
		"""
		Getter for attribute **id**.

		Returns
		-------
		: int
			The node's id.

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
			The node's id.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.id = id

	#--------------------------------------------------------------------------

	def get_father(self):
		"""
		Getter for attribute **father**.

		Returns
		-------
		: Any
			The node's father.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.father
	
	#--------------------------------------------------------------------------

	def set_father(self, father) -> None:
		"""
		Setter for attribute **father**.

		father: Any
			The node's father.
		
		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.father = father

	#--------------------------------------------------------------------------

	def get_children(self) -> set:
		"""
		Getter for attribute **children**.

		Returns
		-------
		: set
			The node's set of children nodes.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.children
	
	#--------------------------------------------------------------------------

	def set_children(self, children: set) -> None:
		"""
		Setter for attribute **children**.

		Parameters
		----------
		children: set
			The node's set of children nodes.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.children = children

	#--------------------------------------------------------------------------

	def get_nodes(self) -> set:
		"""
		Getter for attribute **nodes**.

		Returns
		-------
		: set
			The node's set of stored nodes.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.nodes
	
	#--------------------------------------------------------------------------

	def set_nodes(self, nodes: set) -> None:
		"""
		Setter for attribute **nodes**.

		Parameters
		----------
		nodes: set
			The node's set of stored nodes.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.nodes = nodes

	#--------------------------------------------------------------------------

	def get_mser(self) -> float:
		"""
		Getter for attribute **mser**.

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
		Setter for attribute **mser**.

		Returns
		-------
		mser: float
			The node's MSER stability value.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.mser = mser

	#--------------------------------------------------------------------------

	def add_node(self, node) -> None:
		"""
		Adds a stored node.

		Parameters
		----------
		node: Any
			A node to add.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.nodes.add(node)

	#--------------------------------------------------------------------------

	def add_child(self, child) -> None:
		"""
		Adds a child to the node.

		Parameters
		----------
		child: Any
			A child node to add.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.children.add(child)

#------------------------------------------------------------------------------


class Shaping:
	"""
	Implements the concept of Shaping on a MSCT.

	Attributes
	----------
	root: int
		The index of the tree's root node.
	nodes: list
		The list of all nodes.

	Written by Romain PERRIN (<romain.perrin@unistra.fr>).
	"""

	def __init__(self):
		"""
		Constructor - uses its parent initialization.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.root = 0
		self.nodes = []

	#--------------------------------------------------------------------------

	def get_root(self) -> int:
		"""
		Getter for attribute **root**.

		Returns
		-------
		: int
			The tree's root node index.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.root
	
	#--------------------------------------------------------------------------

	def set_root(self, root: int):
		"""
		Setter for attribute **root**.

		Parameters
		----------
		root: int
			The index of the tree's root node.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.root = root

	#--------------------------------------------------------------------------

	def get_nodes(self) -> list:
		"""
		Getter for attribute **nodes**.

		Returns
		-------
		: list
			The list of the tree's nodes.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.nodes
	
	#--------------------------------------------------------------------------

	def set_nodes(self, nodes: list) -> None:
		"""
		Setter for attribute **nodes**.

		Parameters
		----------
		nodes: list
			The list of the tree's nodes.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.nodes = nodes

	#--------------------------------------------------------------------------

	#def get_neighbours(self, index: int, nodes: list[MultiScaleComponentTreeNode], msct_to_index: dict) -> list:
	def get_neighbours(self, index: int, nodes: list[MSERNode], msct_to_index: dict) -> list:
		"""
		Returns a list of indices of neighbour nodes of **node** according the MSCT parenthood relation.

		Parameters
		----------
		index: int
			The index of a node.
		nodes: list
			The list of subtree nodes (valid neighbours are a subset).

		Returns
		-------
		neighbours: list
			The list of indices of neighbouring nodes.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		neighbours = []
		node = nodes[index]
		father = node.get_father()
		if father != node and father in nodes:
			neighbours.append(msct_to_index[father.get_id()])
		for child in node.get_children():
			if child in nodes:
				neighbours.append(msct_to_index[child.get_id()])
		return neighbours

	#--------------------------------------------------------------------------

	def print_tree(self, node: int) -> None:
		"""
		Displays the global structure of the component-tree starting at a given node.

		Parameters
		----------
		node: int
			The index of the root node.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		print(f"# Node ({node}) :")
		print(f"    - MSER   -> {self.get_nodes()[node].get_mser()}")
		print(f"    - Father   -> {self.get_nodes()[node].get_father()}")
		print(f"    - Children   -> [")
		for child in self.get_nodes()[node].get_children():
			print(f"{child}, ")
		print(f"]")

		for child in self.get_nodes()[node].get_children():
			self.print_tree(child)

	#--------------------------------------------------------------------------

	#def build_min_tree(self, tree: MultiScaleComponentTree, root: MultiScaleComponentTreeNode) -> None:
	def build_min_tree(self, tree: MSERTree, root: MSERNode) -> None:
		"""
		Builds a component-tree on a given set of pixels using an implementation based on Najman's algorithm published in :
		L.Najman, M.Croupie, "Building the component-tree in quasi-linear time", Vol. 15, Num. 11, p. 3531-3539, 2006

		Parameters
		----------
		tree: MultiScaleComponentTree
			The MSCT to process.
		root: MultiScaleComponentTreeNode
			The root node of the subtree of tree to process.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		#msct_nodes = list(tree.gather_subtree_nodes(root))
		msct_nodes = list(tree.nodes)
		nb_nodes = len(msct_nodes)

		tree_collection = dict()
		node_collection = dict()
		nodes = dict()
		points = dict()
		neighbours = []
		lowest_node = dict()
		points_to_process = []
		msct_to_index = dict()

		# pre-processing for the two union-find implementations
		for p in range(0, nb_nodes):
			tree_collection[p] = make_set(p)
			node_collection[p] = make_set(p)
			point = ShapingTreePoint()
			point.set_index(p)
			#point.set_value(msct_nodes[p].get_mser())
			point.set_value(msct_nodes[p].get_value())
			points[p] = point
			nodes[p] = make_node(points[p], msct_nodes[p])
			lowest_node[p] = p
			points_to_process.append(points[p])
			msct_to_index[msct_nodes[p].get_id()] = p

		# sort points according to their lexicographical order in increasing order of level
		sorted_points = sorted(points_to_process, key=lambda x: x.get_value(), reverse=False)
		orig = sorted_points[0]

		# main algorithm
		for point in sorted_points:
			p = point.get_index()

			# search for the canonical node corresponding to the point p
			cur_tree = find(tree_collection, p)
			cur_node = find(node_collection, lowest_node[cur_tree])

			neighbours = self.get_neighbours(p, msct_nodes, msct_to_index)

			# for each neighbour in the 4-neighbourhood
			for q in neighbours:

				# if the neighbour has already been processed
				#if (msct_nodes[q].get_mser() < msct_nodes[p].get_mser()) or (msct_nodes[q].get_mser() == msct_nodes[p].get_mser() and q < p):
				if (msct_nodes[q].get_value() < msct_nodes[p].get_value()) or (msct_nodes[q].get_value() == msct_nodes[p].get_value() and q < p):

					# search for the canonical node corresponding to the point q
					adj_tree = find(tree_collection, q)
					adj_node = find(node_collection, lowest_node[adj_tree])

					# if the two points are not already in the same node
					if (cur_node != adj_node):

						# if the two canonical nodes have the same level
						# it means that these two nodes are in fact part of the same component
						if (nodes[cur_node].get_mser() == nodes[adj_node].get_mser()):
							# merge the two nodes
							cur_node = merge_nodes(nodes, node_collection, adj_node, cur_node)

						# the canonical node of q is strictly above the current level
						# it becomes a child of the current node
						else:

							# add to the list of children of the current node
							nodes[cur_node].get_children().add(adj_node)
							nodes[adj_node].set_father(cur_node)

					# link the two partial trees
					cur_tree = link(tree_collection, adj_tree, cur_tree)

					# keep track of the node of lowest level for the union of the two partial trees
					lowest_node[cur_tree] = cur_node

		# root of the component-tree
		#root = lowest_node[TarjanUnionFind.find(tree_collection, TarjanUnionFind.find(node_collection, 0))]
		root = lowest_node[find(tree_collection, find(node_collection, orig.get_index()))]

		# set root and nodes of the component-tree
		self.set_root(root)
		self.set_nodes(nodes)

	#--------------------------------------------------------------------------

	def create_shaping_tree(self) -> MSCTShapingTree:
		"""
		Creates a ShapingTree from the component-tree.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		st = MSCTShapingTree()

		# creating all nodes
		st_nodes = []
		root = None
		for index in range(0, len(self.get_nodes())):
			node = self.get_nodes()[index]
			node_id = node.get_id()
			st_node = MSCTShapingTreeNode()
			st_node.set_id(node_id)
			st_node.set_mser(node.get_mser())
			st_node.set_nodes(node.get_nodes())
			st_nodes.append(st_node)
			if node_id == self.get_root():
				root = st_node

		# father-child relations
		for index in range(0, len(self.get_nodes())):
			node = self.get_nodes()[index]
			node_id = node.get_id()
			father_id = node.get_father()
			st_node = st_nodes[node_id]
			# father-child
			st_node.set_father(st_nodes[father_id])
			st_nodes[father_id].add_child(st_node)
			# child-father
			for child_id in node.get_children():
				st_node.add_child(st_nodes[child_id])
				st_nodes[child_id].set_father(st_node)

		if root in root.get_children():
			root.get_children().remove(root)
		st.set_nodes(set(st_nodes))
		st.set_root(root)

		return st
