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

    def __init__(self):
        self.index = None
        self.value = None

    #----------------------------------------

    def get_index(self) -> int:
        return self.index
    
    def set_index(self, index: int) -> None:
        self.index = index

    #----------------------------------------

    def get_value(self) -> int:
        return self.value

    def set_value(self, value: int) -> None:
        self.value = value

#------------------------------------------------------------------------------

class ShapingTreeNode:

    _idgen = 0

    def __init__(self):
        self.id = ShapingTreeNode._idgen
        ShapingTreeNode._idgen += 1

        self.index = 0
        self.mser = 0
        self.father = 0
        self.children = set()
        self.nodes = set()

    #--------------------------------------------------------------------------

    def get_id(self) -> int:
        return self.id

    def set_id(self, id: int) -> None:
        self.id = id

    #----------------------------------------

    def get_index(self) -> int:
        return self.index

    def set_index(self, index: int) -> None:
        self.index = index

    #----------------------------------------

    def get_mser(self) -> float:
        return self.mser

    def set_mser(self, mser: float) -> None:
        self.mser = mser

    #----------------------------------------

    def get_father(self) -> int:
        return self.father

    def set_father(self, father: int) -> None:
        self.father = father

    #----------------------------------------

    def get_children(self) -> set:
        return self.children

    def set_children(self, children: set) -> None:
        self.children = children

    #----------------------------------------

    def get_nodes(self) -> set:
        return self.nodes

    def set_nodes(self, nodes: set) -> None:
        self.nodes = nodes

    #--------------------------------------------------------------------------

    def add_node(self, node: MultiScaleComponentTreeNode) -> None:
        self.nodes.add(node)

#------------------------------------------------------------------------------

class ShapingTreeSet:

    def __init__(self):
        self.parent = None
        self.rank = None

    #--------------------------------------------------------------------------

    def get_parent(self) -> int:
        return self.parent

    def set_parent(self, parent: int) -> None:
        self.parent = parent

    #----------------------------------------

    def get_rank(self) -> int:
        return self.rank

    def set_rank(self, rank: int) -> None:
        self.rank = rank

#------------------------------------------------------------------------------

def make_node(point: ShapingTreePoint, node: MultiScaleComponentTreeNode) -> ShapingTreeNode:
    n = ShapingTreeNode()
    n.set_index(point.get_index())
    n.set_mser(point.get_value())
    n.set_father(point.get_index())
    n.set_children(set())
    n.add_node(node)
    return n

#------------------------------------------------------------------------------

def make_set(x: int) -> ShapingTreeSet:
    '''
    Add the set {x} to the collection Q, provided that the element x does not already belongs to a set in Q.

            Parameters:
                    `x` (int): The set {x}

            Returns:
                    `s` (ComponentTreeSet): the set {x}
    '''
    s = ShapingTreeSet()
    s.set_parent(x)
    s.set_rank(0)
    return s

#------------------------------------------------------------------------------

def find(q: list, x: int) -> int:
    '''
    Return the canonical element of the set in Q which contains x.

            Parameters:
                    `q` (ComponentTreeSet[]): The collection Q
                    `x` (int): The canonical element x

            Returns:
                    find (int): The canonical element of Q containing x
    '''
    if (q[x].get_parent() != x):
        q[x].set_parent(find(q, q[x].get_parent()))

    return q[x].get_parent()

#------------------------------------------------------------------------------

def link(q: list, x: int, y: int) -> int:
    '''
    Let X and Y be the two sets in Q whose canonical elements are x and y respectively (x and y must be different). 
    Both sets are removed from Q, their union Z = X ∪ Y is added to Q and a canonical element for Z is selected and returned.

            Parameters:
                    q (ComponentTreeSet[]): The collection Q
                    x (int): The canonical element x
                    y (int): The canonical element y

            Returns:
                    y (int): The canonical element of Z = X ∪ Y
    '''
    if (q[x].get_rank() > q[y].get_rank()):
        x, y = y, x
        
    if (q[x].get_rank() == q[y].get_rank()):
        q[y].set_rank(q[y].get_rank() + 1)

    q[x].set_parent(y)
    return y

#------------------------------------------------------------------------------


def merge_nodes(nodes: list[ShapingTreeNode], q: ShapingTreeSet, n1: int, n2: int) -> int:
    '''
    Merge two nodes (attributes, etc.) together and return the index of the resulting node.

            Parameters:
                    `nodes` (ShapingTreeNode[]): The list of nodes
                    `q` (ComponentTreeSet): The collection Q
                    `n1` (int): Index of the first node
                    `n2` (int): Index of the second node

            Returns:
                    `tmp_n1` (int): index of the resulting node
    '''
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

    _idgen = 0

    def __init__(self):
        self.id = MSCTShapingTree._idgen
        MSCTShapingTree._idgen += 1
        self.root = None
        self.nodes = set()

    #----------------------------------------

    def get_root(self) -> ShapingTreeNode:
        return self.root
    
    def set_root(self, root: ShapingTreeNode) -> None:
        self.root = root

    #----------------------------------------

    def get_nodes(self) -> set[ShapingTreeNode]:
        return self.nodes
    
    def set_nodes(self, nodes: set[ShapingTreeNode]) -> None:
        self.nodes = nodes

    #----------------------------------------

    def add_node(self, node: ShapingTreeNode) -> None:
        self.get_nodes().add(node)

    def remove_node(self, node: ShapingTreeNode) -> None:
        self.get_nodes().remove(node)

    #----------------------------------------

    def filter_return_leq(self, mser: float) -> set[MSERNode]:
        '''
        Returns only nodes of the tree whose MSER value is lesser or equal to `mser`.
        MSCT nodes contained are sreturned instead of MSCTShapingTreeNode
        '''
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
        '''
        Saves the current tree to a dot file using the DOT language and pyDot/GraphViz.

                Parameters:
                        `filename` (str): path to the output file without extension
                        `debug` (bool): whether to print debug informations

                Returns:
                        None
        '''

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

    _idgen = 0

    def __init__(self):
        self.id = MSCTShapingTreeNode._idgen
        MSCTShapingTreeNode._idgen += 1

        self.father = None
        self.children = set()
        self.nodes = set()
        self.mser = 0.0

    #--------------------------------------------------------------------------

    def get_id(self) -> int:
        return self.id

    def set_id(self, id: int) -> None:
        self.id = id

    #----------------------------------------

    def get_father(self):
        return self.father

    def set_father(self, father) -> None:
        self.father = father

    #----------------------------------------

    def get_children(self) -> set:
        return self.children

    def set_children(self, children: set) -> None:
        self.children = children

    #----------------------------------------

    def get_nodes(self) -> set:
        return self.nodes

    def set_nodes(self, nodes: set) -> None:
        self.nodes = nodes

    #----------------------------------------

    def get_mser(self) -> float:
        return self.mser

    def set_mser(self, mser: float) -> None:
        self.mser = mser

    #----------------------------------------

    def add_node(self, node) -> None:
        self.nodes.add(node)

    #----------------------------------------

    def add_child(self, child) -> None:
        self.children.add(child)

#------------------------------------------------------------------------------


class Shaping:

    def __init__(self):
        '''
        Constructor - uses its parent initialization.
        '''
        self.root = 0
        self.nodes = []

    #----------------------------------------

    def get_root(self) -> int:
        '''
        Getter for attribute `root`
        '''
        return self.root

    def set_root(self, root: int):
        '''
        Setter for attribute `root`
        '''
        self.root = root

    #----------------------------------------

    def get_nodes(self) -> list:
        '''
        Getter for attribute `nodes`
        '''
        return self.nodes

    def set_nodes(self, nodes: list) -> None:
        '''
        Setter for attribute `nodes`
        '''
        self.nodes = nodes

    #----------------------------------------

    #def get_neighbours(self, index: int, nodes: list[MultiScaleComponentTreeNode], msct_to_index: dict) -> list:
    def get_neighbours(self, index: int, nodes: list[MSERNode], msct_to_index: dict) -> list:
        '''
        Returns a list of indices of neighbour nodes of `node` according the MSCT parenthood relation.

                Parameters:
                        `index` (int): index of a node
                        `nodes` (list): list of subtree nodes (valid neighbours are a subset)

                Returns:
                        `neighbours` (list): list of indices of neighbouring nodes
        '''
        neighbours = []
        node = nodes[index]
        father = node.get_father()
        if father != node and father in nodes:
            neighbours.append(msct_to_index[father.get_id()])
        for child in node.get_children():
            if child in nodes:
                neighbours.append(msct_to_index[child.get_id()])
        return neighbours

    #----------------------------------------

    def print_tree(self, node: int) -> None:
        '''
        Displays the global structure of the component-tree starting at a given node.

                Parameters:
                        `node` (int): Index of the root node
        '''
        print(f"# Node ({node}) :")
        print(f"    - MSER   -> {self.get_nodes()[node].get_mser()}")
        print(f"    - Father   -> {self.get_nodes()[node].get_father()}")
        print(f"    - Children   -> [")
        for child in self.get_nodes()[node].get_children():
            print(f"{child}, ")
        print(f"]")

        for child in self.get_nodes()[node].get_children():
            self.print_tree(child)

    #----------------------------------------

    #def build_min_tree(self, tree: MultiScaleComponentTree, root: MultiScaleComponentTreeNode) -> None:
    def build_min_tree(self, tree: MSERTree, root: MSERNode) -> None:
        '''
        Builds a component-tree on a given set of pixels using an implementation based on Najman's algorithm published in :
        L.Najman, M.Croupie, "Building the component-tree in quasi-linear time", Vol. 15, Num. 11, p. 3531-3539, 2006

                Parameters:
                        `tree` (MultiScaleComponentTree): MSCT to process
                        `root` (MultiScaleComponentTreeNode): root node of the subtree of `tree` to proecess

                Returns:
                        None
        '''
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
        '''
        Creates a ShapingTree from the component-tree
        '''
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
