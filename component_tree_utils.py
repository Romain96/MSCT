#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import math
from component_tree_point import ComponentTreePoint
from component_tree_set import ComponentTreeSet
from component_tree_node import ComponentTreeNode
from tarjan_union_find import TarjanUnionFind

#------------------------------------------------------------------------------

def make_node(point: ComponentTreePoint) -> ComponentTreeNode:
    '''
    Creates a node the size of a point.

            Parameters:
                    `point` (ComponentTreePoint): A point (i.e. a pixel)

            Returns:
                    `n` (ComponentTreeNode): A new node containing the given point
    '''
    n = ComponentTreeNode()
    n.set_index(point.get_index())
    n.set_level(point.get_value())
    n.set_highest(point.get_value())
    n.set_area(1)
    n.set_subarea(1)
    n.set_father(point.get_index())
    n.set_children(set())
    n.get_pixels().add(point.get_index())
    return n

#------------------------------------------------------------------------------

def merge_nodes(nodes: list, q: ComponentTreeSet, n1: int, n2: int) -> int:
    '''
    Merge two nodes (attributes, etc.) together and return the index of the resulting node.

            Parameters:
                    `nodes` (ComponentTreeNode[]): The list of nodes
                    `q` (ComponentTreeSet): The collection Q
                    `n1` (int): Index of the first node
                    `n2` (int): Index of the second node

            Returns:
                    `tmp_n1` (int): index of the resulting node
    '''
    tmp_n1 = TarjanUnionFind.link(q, n1, n2)
    if tmp_n1 == n2:
        tmp_n2 = n1
    else:
        tmp_n2 = n2

    # update attributes
    nodes[tmp_n1].set_highest(max(nodes[tmp_n1].get_highest(), nodes[tmp_n2].get_highest()))
    nodes[tmp_n1].set_area(nodes[tmp_n1].get_area() + nodes[tmp_n2].get_area())
    nodes[tmp_n1].get_pixels().update(nodes[tmp_n2].get_pixels())
    nodes[tmp_n2].get_pixels().clear()
    nodes[tmp_n1].set_subarea(nodes[tmp_n1].get_subarea() + nodes[tmp_n2].get_subarea())

    # add the list of children of the node that is not kept to the list of children of the node that is kept
    for child in nodes[tmp_n2].get_children():
        nodes[tmp_n1].get_children().add(child)
        nodes[child].set_father(tmp_n1)

    nodes[tmp_n2].set_children([])
    return tmp_n1

#------------------------------------------------------------------------------

def get_neighbours(x: int, ny: int, nx: int) -> list:
    '''
    Retrieves the 4-neighbourdhood of the point of index x.

            Parameters:
                    `x` (int): Index of the point
                    `ny` (int): Number of rows
                    `nx` (int): Number of columns

            Returns:
                    `neighbours` (int[]): list of indices representing the neighbouring points of x
    '''
    neighbours = []

    # if the point is not on the very top of the image
    if math.floor(x / nx) > 0:
        neighbours.append(x - nx)

    # if the point is not on the very left of the image
    if x % nx > 0:
        neighbours.append(x - 1)

    # if the point is not on the very right of the image
    if x % nx < nx - 1:
        neighbours.append(x + 1)

    # if the point is not on the very bottom of the image
    if math.floor(x / nx) < ny - 1:
        neighbours.append(x + nx)

    return neighbours

#------------------------------------------------------------------------------

def lex_sort(a: ComponentTreePoint, b: ComponentTreePoint) -> bool:
    '''
    Lexicographical order comparison in decreasing order of level between points.
    For example : (12, 120) < (3, 50) < (4, 50) < (6, 40)

            Parameters:
                    `a` (Point): The point A
                    `b` (Point): The point B

            Returns:
                    `lex_sort` (bool): A > B by decresing lexicographical order
    '''
    if a.value() > b.value():
        return True
    if b.value() > a.value():
        return False
    return a.index() < b.index()