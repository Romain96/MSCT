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
    Represents a node of a multiscale component-tree (MSCT)
    """

    _id_generator = 0
        
    def __init__(self):
        '''
        Constructor - generates a unique ID, initializes all integers to 0 and sets to empty
        '''
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
        '''
        Getter for attribute `id`
        '''
        return self.id

    def set_id(self, id: int) -> None:
        '''
        Setter for attribute `id`
        '''
        self.id = id

    #----------------------------------------

    def get_scales(self) -> list[int]:
        '''
        Returns a list of the current nodes' scales
        '''
        return self.pixels.keys()

    #----------------------------------------

    def get_level(self) -> int:
        '''
        Getter for attribute `level`
        '''
        return self.level

    def set_level(self, level: int) -> None:
        '''
        Setter for attribute `level`
        '''
        self.level = level

    #----------------------------------------

    def get_father(self) -> Self:
        '''
        Getter for attribute `father`
        '''
        return self.father

    def set_father(self, father) -> None:
        '''
        Setter for attribute `father`
        '''
        self.father = father

    #----------------------------------------

    def get_children(self) -> set[Self]:
        '''
        Getter for attribute `children`
        '''
        return self.children

    def set_children(self, children: set) -> None:
        '''
        Setter for attribute `children`
        '''
        self.children = children

    #----------------------------------------

    def get_pixels(self) -> dict:
        '''
        Getter for attribute `pixels`
        '''
        return self.pixels

    def set_pixels(self, pixels: dict) -> None:
        '''
        Setter for attribute `pixels`
        '''
        self.pixels = pixels

    #----------------------------------------

    def get_pixels_at_scale(self, scale: int) -> set:
        '''
        Returns the set of pixels contained in the node at the given scale.
        '''
        if scale in self.pixels.keys():
            return self.pixels[scale]
        raise Exception(f"The given scale {scale} does not exist in the current node {self.id} (scales = {self.pixels.keys()})")

    def set_pixels_at_scale(self, pixels: set, scale: int) -> None:
        '''
        Replaces the set of pixels at the given scale by the given set if it exists or creates it otherwise
        '''
        self.pixels[scale] = pixels

    def merge_pixels_at_scale(self, pixels: set, scale: int) -> None:
        '''
        Merges the given set of pixels at the given scale with the existing one.
        '''
        if scale in self.get_scales():
            merged_pixels = self.get_pixels_at_scale(scale)
            merged_pixels.update(pixels)
            self.set_pixels_at_scale(merged_pixels, scale)
        else:
            self.set_pixels_at_scale(pixels, scale)

    #----------------------------------------

    def get_area(self) -> int:
        '''
        Returns the pixel area of a node at the global scale.
        '''
        return self.area

    def set_area(self, area: int) -> None:
        '''
        Sets the pixel area of a node at the global scale.
        '''
        self.area = area

    def compute_area(self) -> None:
        '''
        Computes the global scale area of a node using its pixels dictionnary.
        Sets the `area` attribute of the node.
        '''
        area = 0
        for key in self.pixels.keys():
            area += pow(4, key) * len(self.get_pixels_at_scale(scale=key))
        self.set_area(area)

    #----------------------------------------

    def get_subarea(self) -> int:
        '''
        Returns the pixel area of a subtree excluding the current node at the global scale.
        '''
        return self.subarea

    def set_subarea(self, subarea: int) -> None:
        '''
        Sets the pixel area of a subtree excluding the current node at the global scale.
        '''
        self.subarea = subarea

    #----------------------------------------

    def get_mser(self) -> float:
        '''
        Getter for attribute `mser` (Maximally Stable Extremal Regions)
        '''
        return self.mser

    def set_mser(self, mser: float) -> None:
        '''
        Setter for attribute `mser` (Maximally Stable Extremal Regions)
        '''
        self.mser = mser

    #----------------------------------------

    def get_active(self) -> bool:
        '''
        Getter for attribute `active`
        '''
        return self.active

    def set_active(self, active: bool) -> None:
        '''
        Setter for attribute `active`
        '''
        self.active = active

    #----------------------------------------

    def get_histogram(self) -> list[int]:
        '''
        Getter for attribute `histogram`
        '''
        return self.histogram
    
    def set_histogram(self, histogram: list[int]) -> None:
        '''
        Setter for attribute `histogram`
        '''
        self.histogram = histogram

    #--------------------------------------------------------------------------

    def add_child(self, child) -> None:
        '''
        Adds a child to the set of children of the node

                Parameters:
                        `child` (MultiScaleComponentTreeNode): child node to add

                Returns:
                        None
        '''
        self.children.add(child)

    #--------------------------------------------------------------------------

    def remove_child(self, child) -> None:
        '''
        Removed a child to the set of children of the node

                Parameters:
                        `child` (MultiScaleComponentTreeNode): child node to remove

                Returns:
                        None
        '''
        self.children.remove(child)

    #--------------------------------------------------------------------------

    def get_total_area(self) -> int:
        '''
        Returns the area of the node plus the area of its subtree at the global scale

                Parameters:
                        None

                Returns:
                        (int) the total area of the node in pixels
        '''
        return self.get_area() + self.get_subarea()

    #--------------------------------------------------------------------------

    def is_leaf(self) -> bool:
        '''
        Returns True if the node is a leaf (doesn't have any child), False otherwise

                Parameters:
                        None

                Returns:
                        (bool)
        '''
        if len(self.get_children()) == 0:
            return True
        else:
            return False
        
    #--------------------------------------------------------------------------

    def is_descendent_of(self, node) -> bool:
        '''
        Returns True if the node is a descendent of the node given in parameter, False otherwise

                Parameters:
                        `node` (MultiScaleComponentTreeNode): other node for comparison

                Returns:
                        (bool)
        '''
        if self.get_father() == self:
            return False
        elif self.get_father() == node:
            return True
        else:
            return self.get_father().is_descendent_of(node)
        
    #--------------------------------------------------------------------------

    def is_ascendent_of(self, node) -> bool:
        '''
        Returns True if the node is a ascendent of the node given in parameter, False otherwise

                Parameters:
                        `node` (MultiScaleComponentTreeNode): other node for comparison

                Returns:
                        (bool)
        '''
        if node.get_father() == node.get_father():
            return False
        elif node.get_father() == self:
            return True
        else:
            return self.is_ascendent_of(node.get_father())
    
