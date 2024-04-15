#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

#------------------------------------------------------------------------------

class ComponentTreeSet():
    """
    A class devoted to the component-tree construction, it represents a basic set.

    ...

    Attributes
    ----------
    parent : int
        parent of the set as index
    rank : int
        rank of the set [0,+inf[
    """

    def __init__(self):
        '''
        Constructor - it sets parent and rank to None
        '''
        self.parent = None # parent of the canonical element
        self.rank = None   # rank of the canonical element

    #--------------------------------------------------------------------------

    def get_parent(self) -> int:
        '''
        Getter for attribute `parent`
        '''
        return self.parent

    def set_parent(self, parent: int) -> None:
        '''
        Setter for attribute `parent`
        '''
        self.parent = parent

    #----------------------------------------

    def get_rank(self) -> int:
        '''
        Getter for attribute `rank`
        '''
        return self.rank

    def set_rank(self, rank: int) -> None:
        '''
        Setter for attribute `rank`
        '''
        self.rank = rank
