#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import numpy as np

from component_tree_point import ComponentTreePoint
from component_tree import ComponentTree
from tarjan_union_find import TarjanUnionFind
from component_tree_utils import make_node, merge_nodes, get_neighbours

#------------------------------------------------------------------------------

class MaxTree(ComponentTree):
    """
    Represents a max-tree, with a root node and a list of nodes (ComponentTreeNode).

    ...

    Attributes
    ----------
    inherited ComponentTree attributes

    Methods
    -------
    build_component_tree(image, invert, debug):
        builds the component-tree of the given image using Najman's algorithm
    build_component_tree_from_partial_image(image, mask, invert, debug):
        builds the component-tree of a partial image using Najman's algorithm
    inherited ComponentTree methods
    
    """

    def __init__(self):
        '''
        Constructor - uses its parent initialization.
        '''
        super().__init__()

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
        ny = image.shape[0] # number of rows
        nx = image.shape[1] # number of columns
        nn = ny * nx        # number of points

        image_ptr = image.flatten()

        if invert_image:
            self.set_invert(True)
            image_ptr = 255 - image_ptr
        else:
            self.set_invert(False)

        tree_collection = []
        node_collection = []
        nodes = []
        points = []
        neighbours = []
        
        # auxiliary map which associates, to each canonical element of
        # tree_collection, the root of the corresponding partial tree
        lowest_node = [0] * nn

        # pre-processing for the two union-find implementations
        for p in range(0, nn):
            tree_collection.append(TarjanUnionFind.make_set(p))
            node_collection.append(TarjanUnionFind.make_set(p))

            point = ComponentTreePoint()
            point.set_index(p)
            point.set_value(image_ptr[p])
            points.append(point)
            nodes.append(make_node(point))
            lowest_node[p] = p

        # sort points according to their lexicographical order in decreasing order of level
        sorted_points = sorted(points, key=lambda x: x.get_value(), reverse=True)

        # main algorithm
        for point in sorted_points:
            p = point.get_index()

            # search for the canonical node corresponding to the point p
            cur_tree = TarjanUnionFind.find(tree_collection, p)
            cur_node = TarjanUnionFind.find(node_collection, lowest_node[cur_tree])

            neighbours = get_neighbours(p, ny, nx)

            # for each neighbour in the 4-neighbourhood
            for q in neighbours:

                # if the neighbour has already been processed
                if (image_ptr[q] > image_ptr[p]) or (image_ptr[q] == image_ptr[p] and q < p):

                    # search for the canonical node corresponding to the point q
                    adj_tree = TarjanUnionFind.find(tree_collection, q)
                    adj_node = TarjanUnionFind.find(node_collection, lowest_node[adj_tree])

                    # if the two points are not already in the same node
                    if (cur_node != adj_node):

                        # if the two canonical nodes have the same level
                        # it means that these two nodes are in fact part of the same component
                        if (nodes[cur_node].get_level() == nodes[adj_node].get_level()):
                            # merge the two nodes
                            cur_node = merge_nodes(nodes, node_collection, adj_node, cur_node)

                        # the canonical node of q is strictly above the current level
                        # it becomes a child of the current node
                        else:

                            # update attributes
                            nodes[cur_node].set_highest(max(nodes[cur_node].get_highest(), nodes[adj_node].get_highest()))
                            nodes[cur_node].set_area(nodes[cur_node].get_area() + nodes[adj_node].get_area())
                            nodes[cur_node].set_subarea(nodes[cur_node].get_subarea() + nodes[adj_node].get_subarea())
                            # add to the list of children of the current node
                            nodes[cur_node].get_children().add(adj_node)
                            nodes[adj_node].set_father(cur_node)

                    # link the two partial trees
                    cur_tree = TarjanUnionFind.link(tree_collection, adj_tree, cur_tree)

                    # keep track of the node of lowest level for the union of the two partial trees
                    lowest_node[cur_tree] = cur_node

        # root of the component-tree
        root = lowest_node[TarjanUnionFind.find(tree_collection, TarjanUnionFind.find(node_collection, 0))]

        # set root and nodes of the component-tree
        self.set_root(root)
        self.set_nodes(nodes)

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
        ny = image.shape[0] # number of rows
        nx = image.shape[1] # number of columns
        nn = ny * nx        # number of points

        image_ptr = image.flatten()
        mask_ptr = mask.flatten()

        if invert_image:
            self.set_invert(True)
            image_ptr = 255 - image_ptr
        else:
            self.set_invert(False)

        tree_collection = dict()
        node_collection = dict()
        nodes = dict()
        points = dict()
        neighbours = []
        lowest_node = dict()
        points_to_process = []

        # pre-processing for the two union-find implementations
        for p in range(0, nn):
            # only for activated pixels in the mask
            if mask_ptr[p]:
                tree_collection[p] = TarjanUnionFind.make_set(p)
                node_collection[p] = TarjanUnionFind.make_set(p)
                point = ComponentTreePoint()
                point.set_index(p)
                point.set_value(image_ptr[p])
                points[p] = point
                nodes[p] = make_node(points[p])
                lowest_node[p] = p
                points_to_process.append(points[p])

        # sort points according to their lexicographical order in decreasing order of level
        sorted_points = sorted(points_to_process, key=lambda x: x.get_value(), reverse=True)
        orig = sorted_points[0]

        # main algorithm
        for point in sorted_points:
            p = point.get_index()
            if not mask_ptr[p]:
                continue

            # search for the canonical node corresponding to the point p
            cur_tree = TarjanUnionFind.find(tree_collection, p)
            cur_node = TarjanUnionFind.find(node_collection, lowest_node[cur_tree])

            neighbours = get_neighbours(p, ny, nx)

            # for each neighbour in the 4-neighbourhood
            for q in neighbours:
                if not mask_ptr[q]:
                    continue

                # if the neighbour has already been processed
                if (image_ptr[q] > image_ptr[p]) or (image_ptr[q] == image_ptr[p] and q < p):

                    # search for the canonical node corresponding to the point q
                    adj_tree = TarjanUnionFind.find(tree_collection, q)
                    adj_node = TarjanUnionFind.find(node_collection, lowest_node[adj_tree])

                    # if the two points are not already in the same node
                    if (cur_node != adj_node):

                        # if the two canonical nodes have the same level
                        # it means that these two nodes are in fact part of the same component
                        if (nodes[cur_node].get_level() == nodes[adj_node].get_level()):
                            # merge the two nodes
                            cur_node = merge_nodes(nodes, node_collection, adj_node, cur_node)

                        # the canonical node of q is strictly above the current level
                        # it becomes a child of the current node
                        else:

                            # update attributes
                            nodes[cur_node].set_highest(max(nodes[cur_node].get_highest(), nodes[adj_node].get_highest()))
                            nodes[cur_node].set_area(nodes[cur_node].get_area() + nodes[adj_node].get_area())
                            nodes[cur_node].set_subarea(nodes[cur_node].get_subarea() + nodes[adj_node].get_subarea())
                            
                            # add to the list of children of the current node
                            nodes[cur_node].get_children().add(adj_node)
                            nodes[adj_node].set_father(cur_node)

                    # link the two partial trees
                    cur_tree = TarjanUnionFind.link(tree_collection, adj_tree, cur_tree)

                    # keep track of the node of lowest level for the union of the two partial trees
                    lowest_node[cur_tree] = cur_node

        # root of the component-tree
        #root = lowest_node[TarjanUnionFind.find(tree_collection, TarjanUnionFind.find(node_collection, 0))]
        root = lowest_node[TarjanUnionFind.find(tree_collection, TarjanUnionFind.find(node_collection, orig.get_index()))]

        # set root and nodes of the component-tree
        self.set_root(root)
        self.set_nodes(nodes)
