#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

#------------------------------------------------------------------------------

class ComponentTreePoint():
    """
    A class devoted to the component-tree construction, it represents a basic point (pixel).

    ...

    Attributes
    ----------
    index : int
        index of the pixel (x,y coordinates in 1D)
    value : int
        gray value of the pixel [0,255]
    """

    def __init__(self):
        '''
        Constructor - it sets index and value to None
        '''
        self.index = None # index of the point
        self.value = None # gray value of the point

    #--------------------------------------------------------------------------

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

    def get_value(self) -> int:
        '''
        Getter for attribute `value`
        '''
        return self.value

    def set_value(self, value: int) -> None:
        '''
        Setter for attribute `value`
        '''
        self.value = value
