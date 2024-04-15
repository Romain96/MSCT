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

    ...

    Attributes
    ----------
    root : int
        index of the root node
    nodes : list
        list of nodes (ComponentTreeNode)
    invert : bool
        whether the image has to be inverted (default if bright objects on dark background)

    Methods
    -------
    print_tree(node):
        print the tree structure in the console starting at the given node
    save_dot(filename):
        writes the component-tree in the DOT language into a dot file
    build_component_tree(image, invert):
        builds the component-tree of the given image using Najman's algorithm
    build_component_tree_from_partial_image(image, mask, invert):
        builds the component-tree of a partial image using Najman's algorithm
    reconstruct_image(image):
        reconstructs an image using the component-tree
    """

    def __init__(self):
        '''
        Constructor - initializes the root to 0, creates an empty list of nodes and sets invert to False.
        '''
        self.root = 0 # index of the root
        self.nodes = list()  # list of nodes
        self.invert = False

    #--------------------------------------------------------------------------

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

    def get_invert(self) -> bool:
        '''
        Getter for attribute `invert`
        '''
        return self.invert

    def set_invert(self, invert: bool) -> None:
        '''
        Setter for attribute `invert`
        '''
        self.invert = invert

    #--------------------------------------------------------------------------

    def print_tree(self, node: int) -> None:
        '''
        Displays the global structure of the component-tree starting at a given node.

                Parameters:
                        `node` (int): Index of the root node
        '''
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
        '''
        Saves the current component-tree to a dot file using the DOT language and pyDot/GraphViz.

                Parameters:
                        `filename` (str): path to the output file without extension

                Returns:
                        None
        '''

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
        '''
        Builds a component-tree using an implementation based on Najman's algorithm published in :
        L.Najman, M.Croupie, "Building the component-tree in quasi-linear time", Vol. 15, Num. 11, p. 3531-3539, 2006

                Parameters:
                        `image` (nparray): Numpy array of the image
                        `invert_image` (bool): whether the input image should be inverted

                Returns:
                        None
        '''
        # placeholder, implementation in derived classes
        pass

    #--------------------------------------------------------------------------

    def build_component_tree_from_partial_image(self, image: np.ndarray, mask:np.ndarray, invert_image: bool) -> None:
        '''
        Builds a component-tree on a given set of pixels using an implementation based on Najman's algorithm published in :
        L.Najman, M.Croupie, "Building the component-tree in quasi-linear time", Vol. 15, Num. 11, p. 3531-3539, 2006

                Parameters:
                        `image` (nparray): Numpy array of the image
                        `mask` (ndarray): Numpy array of the same size as image, indicating which pixels should be processed
                        `invert_image` (bool): whether the input image should be inverted

                Returns:
                        None
        '''
        # placeholder, implementation in derived classes
        pass

    #--------------------------------------------------------------------------

    def reconstruct_image(self, image: np.ndarray) -> np.ndarray:
        '''
        Reconstructs the image using the component-tree.

                Parameters:
                        `image` (ndarray): Numpy array of the original image

                Returns:
                        `reconstructed` (ndarray): Numpy array of the reconstructed image
        '''
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
