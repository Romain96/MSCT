#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import numpy as np

#------------------------------------------------------------------------------

class CellGraphNode:

    def __init__(self, type: str, pixels: set):
        self.type = type
        self.pixels = pixels

    def get_type(self) -> str:
        return self.type

    def set_type(self, type: str) -> None:
        self.type = type

    def get_pixels(self) -> set:
        return self.pixels
    
    def set_pixels(self, pixels: set) -> None:
        self.pixels = pixels

    def reset(self) -> None:
        self.type = None
        self.pixels.clear()

#------------------------------------------------------------------------------

class CellGraph:

    def __init__(self):
        self.root = None
        self.nodes = set()

    def get_root(self) -> CellGraphNode:
        return self.root
    
    def set_root(self, root: CellGraphNode) -> None:
        self.root = root

    def get_nodes(self) -> set:
        return self.nodes

    def set_nodes(self, nodes: set) -> None:
        self.nodes = nodes

    def add_node(self, node: CellGraphNode) -> None:
        self.nodes.add(node)

    def remove_node(self, node: CellGraphNode) -> None:
        self.nodes.remove(node)

    def reset(self) -> None:
        self.root = None
        self.nodes.clear()