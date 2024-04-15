#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

from component_tree_set import ComponentTreeSet

#------------------------------------------------------------------------------

class TarjanUnionFind():
    """
    Class containing methods to manage a set of disjointed sets using Tarjan's union-find algorithm
    """

    def __init__(self):
        pass

    #--------------------------------------------------------------------------

    def make_set(x: int) -> ComponentTreeSet:
        '''
        Add the set {x} to the collection Q, provided that the element x does not already belongs to a set in Q.

                Parameters:
                        `x` (int): The set {x}

                Returns:
                        `s` (ComponentTreeSet): the set {x}
        '''
        s = ComponentTreeSet()
        s.set_parent(x)
        s.set_rank(0)
        return s

    #--------------------------------------------------------------------------

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
            q[x].set_parent(TarjanUnionFind.find(q, q[x].get_parent()))

        return q[x].get_parent()

    #--------------------------------------------------------------------------

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