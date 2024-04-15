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

    ...

    Attributes
    ----------
    id : int
        unique ID of the node
    index : int
        index of the node
    level : int
        gray level of the node
    highest : int
        highest gray level of the node
    area : int
        pixel area of the node
    father : int
        index of the father node
    children : set
        set of indices of children nodes
    pixels : set
        set of indices of pixels (ComponentTreePoint)
    """

    _id_generator = 0

    def __init__(self):
        '''
        Constructor - initializes the unique ID, initializes all remaining integers to 0 and sets to empty.
        '''
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

    def get_index(self) -> int:
        '''
        Getter for attribute `index`
        '''
        return self.index

    def set_index(self, index: int) -> None:
        '''
        Setter for attribute `index`
        '''
        self.index = index

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

    def get_highest(self) -> int:
        '''
        Getter for attribute `highest`
        '''
        return self.highest

    def set_highest(self, highest: int) -> None:
        '''
        Setter for attribute `highest`
        '''
        self.highest = highest

    #----------------------------------------
    
    def get_area(self) -> int:
        '''
        Getter for attribute `area`
        '''
        return self.area

    def set_area(self, area: int) -> None:
        '''
        Setter for attribute `area`
        '''
        self.area = area

    #----------------------------------------

    def get_subarea(self) -> int:
        '''
        Getter for attribute `subarea`
        '''
        return self.subarea
    
    def set_subarea(self, subarea: int) -> None:
        '''
        Setter for attribute `subarea`
        '''
        self.subarea = subarea

    #----------------------------------------

    def get_father(self) -> int:
        '''
        Getter for attribute `father`
        '''
        return self.father

    def set_father(self, father: int) -> None:
        '''
        Setter for attribute `father`
        '''
        self.father = father

    #----------------------------------------

    def get_children(self) -> set:
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

    def get_pixels(self) -> set:
        '''
        Getter for attribute `pixels`
        '''
        return self.pixels

    def set_pixels(self, pixels: set) -> None:
        '''
        Setter for attribute `pixels`
        '''
        self.pixels = pixels

    #--------------------------------------------------------------------------

    def add_pixel(self, pixel: int) -> None:
        '''
        Adds a pixel to the node' set of pixels.

            Parameters:
                    `pixel` (int): a pixel represented by its index

            Returns:
                    None
        '''
        self.pixels.add(pixel)
