#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import numpy as np
from multi_scale_component_tree_node import MultiScaleComponentTreeNode
from multi_scale_component_tree import MultiScaleComponentTree

#------------------------------------------------------------------------------

class MSER:
    '''
    Maximally Stable Extremal Regions
    '''

    def init(self):
        pass

    #--------------------------------------------------------------------------

    @staticmethod
    def compute_mser_local_minima_per_branch(
        node: MultiScaleComponentTreeNode, 
        minima: set, 
        local_minima: MultiScaleComponentTreeNode, 
        local_minima_mser: float
    ):
        '''
        Computes a local minimum for each branch of the MSCT.
        The local minima is the node along a branch having the higest stability 
        (i.e. the lowest `stability` attribute value)

                Parameters:
                        `node` (MultiScaleComponentTreeNode): node of the MSCT to process
                        `minima` (set): set of all MSER local minima per branch to fill
                        `local_minima` (MultiScaleComponentTreeNode): current node being the local minima of the current branch
                        `local_minima_mser` (float): MSER value of `local_minima`

                Returns
                        None
        '''
        # case 1 : node is leaf -> adding the current local_minima to the global array minima
        if len(node.get_children()) == 0:
            minima.add(local_minima)

        # case 2 : node has one single child -> recursive call on child down the branch
        elif len(node.get_children()) == 1:
            mser = node.get_mser()

            # case 2-1 : MSER of current node is better than local_minima
            # -> recursive call with local_minima = node and local_minima_mser = node.mser
            if mser < local_minima_mser:
                for child in node.get_children():
                    # 1 child only here
                    MSER.compute_mser_local_minima_per_branch(
                        child, minima=minima, local_minima=node, local_minima_mser=mser
                    )
            # case 2-2 : MSER of current node is worse than local_minima
            # -> recursive call with same local_minima and local_minima_mser
            else:
                for child in node.get_children():
                    # 1 child only here
                    MSER.compute_mser_local_minima_per_branch(
                        child, minima=minima, local_minima=local_minima, local_minima_mser=local_minima_mser
                    )
        
        # case 3 : node has at least 2 children
        # -> adding the local_minima to the global array minima, 
        # -> recursive call on all children with local_minima as themselves
        else:
            mser = node.get_mser()
            if mser < local_minima_mser:
                minima.add(node)
            else:
                minima.add(local_minima)
            for child in node.get_children():
                MSER.compute_mser_local_minima_per_branch(
                    child, minima=minima, local_minima=child, local_minima_mser=child.get_mser()
                )

    #--------------------------------------------------------------------------

    @staticmethod
    def compute_mser_local_minima_per_branch_at_scale(
        scale: int,
        node: MultiScaleComponentTreeNode, 
        minima: set, 
        local_minima: MultiScaleComponentTreeNode, 
        local_minima_mser: float
    ):
        '''
        Computes a local minimum for each branch of the MSCT.
        The local minima is the node along a branch having the higest stability 
        (i.e. the lowest `stability` attribute value).
        Only considers nodes at the given scale `scale`, not every branch might have a local minima !

                Parameters:
                        `node` (MultiScaleComponentTreeNode): node of the MSCT to process
                        `scale` (int): scale at which the MSER nodes will be considered valid candidates
                        `minima` (set): set of all MSER local minima per branch to fill
                        `local_minima` (MultiScaleComponentTreeNode): current node being the local minima of the current branch
                        `local_minima_mser` (float): MSER value of `local_minima`

                Returns
                        None
        '''
        # case 1 : node is leaf -> adding the current local_minima to the global array minima if at the right scale
        if len(node.get_children()) == 0:
            if scale in local_minima.get_scales():
                minima.add(local_minima)

        # case 2 : node has one single child -> recursive call on child down the branch
        elif len(node.get_children()) == 1:
            mser = node.get_mser()
            # case 2-1 : if local_minima is not at the required scale, taking node as the new local_minima
            if scale not in local_minima.get_scales():
                for child in node.get_children():
                    # 1 child only here
                    MSER.compute_mser_local_minima_per_branch_at_scale(
                        scale, child, minima=minima, local_minima=node, local_minima_mser=mser
                    )
            else:
                # case 2-2 : if mser of node is better than mser of local_minima then using node as the new local_minima
                if mser < local_minima_mser:
                    for child in node.get_children():
                        # 1 child only here
                        MSER.compute_mser_local_minima_per_branch_at_scale(
                            scale, child, minima=minima, local_minima=node, local_minima_mser=mser
                        )
                # case 2-3 : if mser of node is worse than mser of local_minima then using the same local_minima
                else:
                    for child in node.get_children():
                        # 1 child only here
                        MSER.compute_mser_local_minima_per_branch_at_scale(
                            scale, child, minima=minima, local_minima=local_minima, local_minima_mser=local_minima_mser
                        )

        # case 3 : node has at least 2 children -> adding the local_minima to the global array minima, recursive call on all children with local_minima as themselves
        else:
            mser = node.get_mser()
            if mser < local_minima_mser:
                if scale in node.get_scales():
                    minima.add(node)
            else:
                if scale in local_minima.get_scales():
                    minima.add(local_minima)
            for child in node.get_children():
                MSER.compute_mser_local_minima_per_branch_at_scale(
                    scale, child, minima=minima, local_minima=child, local_minima_mser=child.get_mser()
                )

#------------------------------------------------------------------------------

    @staticmethod
    def compute_mser_candidates(msct: MultiScaleComponentTree, scale: int, max_area: int, max_mser: float) -> list:
        '''
        Computes a set of candidate nodes to be selected for the enrichment step (scale augmentation) based on MSER values.
        This method uses no user-defined parameter related to min or max areas.

                Parameters:
                        `msct` (MultiScaleComponentTree): MSCT to process
                        `scale` (int): scale at which MSER nodes are considered valid (0 = original scale)
                        `max_area` (int): maximum area allowed
                        `max_mser` (float): maximum MSER value allowed

                Returns:
                        `expanded_msers` (list): list of MSCT nodes to be selected for enrichment
        '''
        all_minima = set()
        MSER.compute_mser_local_minima_per_branch_at_scale(
            scale=scale, node=msct.get_root(), minima=all_minima, 
            local_minima=msct.get_root(), local_minima_mser=msct.get_root().get_mser()
        )

        if msct.get_root() in all_minima:
            all_minima.remove(msct.get_root())
        final_msers = []
        to_process = sorted(list(all_minima), key=lambda x: x.get_mser(), reverse=False)

        while len(to_process) > 0:

            # remove the first element (lowest MSER value)
            element = to_process.pop(0)
            if element.get_total_area() <= max_area:
                # remove all descendants of this element in to_process
                no_descendants = MSER.remove_all_descendants(element, to_process)
                # remove all ancestors of this element in to_process
                no_descendants_no_ancestors = MSER.remove_all_ancestors(element, no_descendants)
                # add the element to final_msers
                final_msers.append(element)
                to_process = no_descendants_no_ancestors

        # removing mser inferior to 1
        filtered_msers = []
        for mser in final_msers:
            if mser.get_mser() <= max_mser:
                filtered_msers.append(mser)

        return filtered_msers
    
#------------------------------------------------------------------------------
    
    @staticmethod
    def compute_mser_candidates_subtree(root: MultiScaleComponentTreeNode, scale: float, max_mser: float) -> list:
        '''
        Computes a set of candidate nodes to be selected for the enrichment step (scale augmentation) based on MSER values.

                Parameters:
                        `root` (MultiScaleComponentTreeNode): subtree root
                        `scale` (int): scale at which MSER nodes are considered valid (0 = original scale)
                        `max_mser` (float): maximum MSER value allowed

                Returns:
                        `expanded_msers` (list): list of MSCT nodes to be selected for enrichment
        '''
        all_minima = set()
        MSER.compute_mser_local_minima_per_branch_at_scale(
            scale=scale, node=root, minima=all_minima, 
            local_minima=root, local_minima_mser=root.get_mser()
        )

        final_msers = []
        to_process = sorted(list(all_minima), key=lambda x: x.get_mser(), reverse=False)

        while len(to_process) > 0:

            # remove the first element (lowest MSER value)
            element = to_process.pop(0)
            # remove all descendants of this element in to_process
            no_descendants = MSER.remove_all_descendants(element, to_process)
            # remove all ancestors of this element in to_process
            no_descendants_no_ancestors = MSER.remove_all_ancestors(element, no_descendants)
            # add the element to final_msers
            final_msers.append(element)
            to_process = no_descendants_no_ancestors

        # removing mser inferior to 1
        filtered_msers = []
        for mser in final_msers:
            if mser.get_mser() <= max_mser:
                filtered_msers.append(mser)

        return filtered_msers
    
#------------------------------------------------------------------------------

    @staticmethod
    def msct_is_ancestor(node_a: MultiScaleComponentTreeNode, node_b: MultiScaleComponentTreeNode) -> bool:
        '''
        Checks whether a node is an ancestor of another node in a MSCT. Nodes should belong to the same MSCT !

                Parameters:
                        `node_a` (MultiScaleComponentTreeNode): MSCT node A
                        `node_b` (MultiScaleComponentTreeNode): MSCT node B

                Returns:
                        (bool): Returns True if `node_a` is an ancestor of `node_b`, False otherwise
        '''
        if node_b.get_father() == node_b:
            return False
        elif node_b.get_father() == node_a:
            return True
        else:
            return MSER.msct_is_ancestor(node_a, node_b.get_father())
    
#------------------------------------------------------------------------------

    @staticmethod
    def msct_is_descendant(node_a: MultiScaleComponentTreeNode, node_b: MultiScaleComponentTreeNode) -> bool:
        '''
        Checks whether a node is a descendant of another node in a MSCT. Nodes should belong to the same MSCT !

                Parameters:
                        `node_a` (MultiScaleComponentTreeNode): MSCT node A
                        `node_b` (MultiScaleComponentTreeNode): MSCT node B

                Returns:
                        (bool): Returns True if `node_a` is a descendant of `node_b`, False otherwise
        '''
        if node_a.get_father() == node_a:
            return False
        elif node_a.get_father() == node_b:
            return True
        else:
            return MSER.msct_is_descendant(node_a.get_father(), node_b)

#------------------------------------------------------------------------------

    @staticmethod
    def remove_all_descendants(node: MultiScaleComponentTreeNode, potential_descendants: list) -> list:
        '''
        Removes all nodes of a list being descendants of a given node.

                Parameters:
                        `node` (MultiScaleComponentTreeNode): MSCT node used as reference
                        `potential_descendants` (list): list of MSCT nodes to check

                Returns:
                        `non_descendants` (list): list of MSCT nodes non descendants of `node`, subset of `potiential_descendants`
        '''
        non_descendants = []
        for candidate in potential_descendants:
            if not MSER.msct_is_descendant(candidate, node):
                non_descendants.append(candidate)
        return non_descendants

#------------------------------------------------------------------------------

    @staticmethod
    def remove_all_ancestors(node: MultiScaleComponentTreeNode, potiential_ancestors: list) -> list:
        '''
        Removes all nodes of a list being descendants of a given node.

                Parameters:
                        `node` (MultiScaleComponentTreeNode): MSCT node used as reference
                        `potiential_ancestors` (list): list of MSCT nodes to check

                Returns:
                        `non_ancestors` (list): list of MSCT nodes non ancestors of `node`, subset of `potiential_ancestors`
        '''
        non_ancestors = []
        for candidate in potiential_ancestors:
            if not MSER.msct_is_ancestor(candidate, node):
                non_ancestors.append(candidate)
        return non_ancestors

