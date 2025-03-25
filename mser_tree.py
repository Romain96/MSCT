#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import pydot

#------------------------------------------------------------------------------

class MSERNode:
	"""
	This class provides a way to represent a node of a MSER tree.

	Written by Romain PERRIN (<romain.perrin@unistra.fr>).
	"""

	def __init__(self, father, value, link) -> None:
		"""
		Constructor of a MSER node.

		Parameters
		----------
		father: Any
			Father of the current node.
		value: Any
			Value to assign to the current node (should be a MSER stability value).
		link: Any
			Type of link to assign to the node.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.id = 0
		self.father = father
		self.children = set()
		self.value = value
		self.link = link

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

		Parameters
		----------
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
		: set[MSERTreeNode]
			Set of the node's children.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.children
	
	#--------------------------------------------------------------------------
	
	def set_children(self, children) -> None:
		"""
		Setter for attribute **children**.

		Parameters
		----------
		children: set[MSERTreeNode]
			Set of the node's children.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.children = children

	#--------------------------------------------------------------------------

	def get_value(self):
		"""
		Getter for attribute **value**.

		Returns
		-------
		: Any
			The node's value.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.value
	
	#--------------------------------------------------------------------------
	
	def set_value(self, value) -> None:
		"""
		Setter for attribute **value**.

		Parameters
		----------
		value: Any
			The node's value.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.value = value

	#--------------------------------------------------------------------------

	def get_link(self):
		"""
		Getter for attribute **link**.

		Returns
		-------
		: Any
			The node's link.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.link
	
	#--------------------------------------------------------------------------
	
	def set_link(self, link) -> None:
		"""
		Setter for attribute **link**.

		Parameters
		----------
		link: Any
			The node's link.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.link = link

	#--------------------------------------------------------------------------

	def add_child(self, child) -> None:
		"""
		Adds a child to the set of the node's children.

		Parameters
		----------
		child: MSERTreeNode
			A node to add to the set of the node's children.

		Returns
		-------
		None

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.children.add(child)

	#--------------------------------------------------------------------------

	def remove_child(self, child) -> None:
		"""
		Removes a node from the set of the node's children.

		Parameters
		----------
		child: MSERTreeNode
			A node to remove from the set of the node's children.

		Returns
		-------
		None

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.children.remove(child)

	#--------------------------------------------------------------------------

	def debug(self) -> None:
		"""
		Prints informations about the node.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		print(f"{MSERNode.__class__.__name__} : ID {hex(id(self))}")
		print(f"{MSERNode.__class__.__name__} : Father {hex(id(self.father))}")
		print(f"{MSERNode.__class__.__name__} : Value {self.value}")
		print(f"{MSERNode.__class__.__name__} : Link {hex(id(self.link))}")

#------------------------------------------------------------------------------

class MSERTree:
	"""
	This class provides a way to represent a MSER Tree.

	Written by Romain PERRIN (<romain.perrin@unistra.fr>).
	"""

	def __init__(self) -> None:
		"""
		Constructor for a MSER Tree.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.root = None
		self.nodes = set()

	#--------------------------------------------------------------------------

	def get_root(self) -> MSERNode:
		"""
		Getter for attribute **root**.

		Returns:
		: MSERNode
			The root node of the MSERTree.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.root
	
	#--------------------------------------------------------------------------

	def set_root(self, root: MSERNode) -> None:
		"""
		Setter for attribute **root**.

		Parameters
		----------
		root: MSERNode
			The root node of the MSERTree.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.root = root

	#--------------------------------------------------------------------------

	def get_nodes(self) -> set[MSERNode]:
		"""
		Getter for attribute **nodes**.

		Returns
		: set[MSERNode]
			The set of all nodes of the MSERTree.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		return self.nodes

	#--------------------------------------------------------------------------

	def set_nodes(self, nodes: set[MSERNode]) -> None:
		"""
		Setter for attribute **nodes**.

		Parameters
		----------
		nodes: set[MSERNode]
			The set of all nodes of the MSERTree.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.nodes = nodes

	#--------------------------------------------------------------------------

	def add_node(self, node: MSERNode) -> None:
		"""
		Adds a node to the set of nodes of the MSERTree.

		Parameters
		----------
		node: MSERNode
			A node to add.
		
		Returns
		-------
		Node

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""
		
		self.nodes.add(node)

	#--------------------------------------------------------------------------

	def remove_node(self, node: MSERNode) -> None:
		"""
		Removes a node from the set of nodes of the MSERTree.

		Parameters
		----------
		node: MSERNode
			A node to remove.

		Returns
		-------
		None

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.nodes.remove(node)

	#--------------------------------------------------------------------------

	def debug(self) -> None:
		"""
		Prints informations about a MSERTree.

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		print(f"{MSERNode.__class__.__name__} : ID {hex(id(self))}")
		print(f"{MSERNode.__class__.__name__} : Root {hex(id(self.root))}")
		print(f"{MSERNode.__class__.__name__} : Nodes {len(self.nodes)}")

	#--------------------------------------------------------------------------

	def save_dot(self, filename: str) -> None:
		"""
		Saves the complete MSERTree in the DOT format to a file.

		Parameters
		----------
		filename: str
			Path to a filename to save the MSERTree without the extension.

		Returns
		-------
		None

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		self.save_dot_from_node(filename, self.root)

	#--------------------------------------------------------------------------

	def save_dot_from_node(self, filename: str, node: MSERNode) -> None:
		"""
		Saves the sub MSERTree rooted in **node** in the DOT format to a file. 

		Parameters
		----------
		filename: str
			Path to a filename to save the MSERTree without the extension.
		node: MSERNode
			The root node representing the subtree that will be saved.
			
		Returns
		-------
		None

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		graph = pydot.Dot('MSERTree', graph_type='graph', bgcolor='white')

		# add all nodes from node to its subtree leaves
		process = set()
		process.add(node)

		while len(process) > 0:
			cur_node = process.pop()
			
			nid = hex(id(cur_node))
			nval = cur_node.value
			label_text = f"id {nid}\n{nval}\n({cur_node.link.get_id()})"
			gnode = pydot.Node(name=f"node_{nid}", label=label_text)
			graph.add_node(gnode)
			
			for child in cur_node.children:
				process.add(child)

		# connect nodes
		process = set()
		process.add(node)

		while len(process) > 0:
			cur_node = process.pop()
			for child in cur_node.children:
				process.add(child)
				nid = hex(id(cur_node))
				cid = hex(id(child))
				edge = pydot.Edge(f"node_{nid}", f"node_{cid}")
				graph.add_edge(edge)

		graph.write_png(f"{filename}.png")

	#--------------------------------------------------------------------------

	def save_dot_highlight_from_node(self, filename: str, node: MSERNode, nodes: set[MSERNode]) -> None:
		"""
		Saves the sub MSERTree rooted in **node** in the DOT format to a file.
		Highlights all nodes present in the subtree that also exists in **nodes**.

		Parameters
		----------
		filename: str
			Path to a filename to save the MSERTRee without the extension.
		node: MSERNode
			The root node representing the subtree that will be saved.
		nodes: set[MSERNode]
			A set of nodes indicating which nodes should be highlighted.

		Returns
		-------
		None

		Written by Romain PERRIN (<romain.perrin@unistra.fr>).
		"""

		graph = pydot.Dot('component_tree', graph_type='graph', bgcolor='white')

		# add all nodes from node to its subtree leaves
		process = set()
		process.add(node)

		while len(process) > 0:
			cur_node = process.pop()

			nid = hex(id(cur_node))
			nval = cur_node.value
			label_text = f"id {nid}\n{nval}\n({cur_node.link.get_id()})"

			if cur_node in nodes:
				gnode = pydot.Node(name=f"node_{cur_node.get_id()}", label=label_text, style='filled', fillcolor='#40e0d0')
				graph.add_node(gnode)
			else:
				gnode = pydot.Node(name=f"node_{cur_node.get_id()}", label=label_text)
				graph.add_node(gnode)

			for child in cur_node.get_children():
				process.add(child)

		# connect nodes
		process = set()
		process.add(node)

		while len(process) > 0:
			cur_node = process.pop()
			for child in cur_node.get_children():
				process.add(child)
				edge = pydot.Edge(f"node_{cur_node.get_id()}", f"node_{child.get_id()}")
				graph.add_edge(edge)

		graph.write_png(f"{filename}.png")

#------------------------------------------------------------------------------
