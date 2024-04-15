#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import numpy as np
#import pydot_ng as pydot
import pydot
from rcc8 import RCC8
from csv import DictWriter
from os.path import join
from sys import maxsize
from typing_extensions import Self
from scipy.ndimage import label

from component_tree import ComponentTree
from component_tree_node import ComponentTreeNode
from max_tree import MaxTree
from multi_scale_component_tree_node import MultiScaleComponentTreeNode
from index_to_pixel import Index2Pixel
from pixel_power_two_scale_converter import PixelPower2ScaleConverter

#------------------------------------------------------------------------------

class MultiScaleComponentTree():
    """
    Represents a multiscale component-tree (MSCT)
    """

    def __init__(self):
        '''
        Constructor - initializes root to None and nodes to an, empty set, creates a pixel converter
        '''
        self.root = None
        self.nodes = set()
        self.invert = False
        self.pconverter = PixelPower2ScaleConverter()

    #--------------------------------------------------------------------------

    def reset(self) -> None:
        '''
        Resets the attributes as the constructor would do
        '''
        self.root = None
        self.nodes = set()
        self.invert = False
        self.pconverter = PixelPower2ScaleConverter()

    #--------------------------------------------------------------------------

    def get_root(self) -> MultiScaleComponentTreeNode:
        '''
        Getter for attribute `root`
        '''
        return self.root
    

    def set_root(self, root: MultiScaleComponentTreeNode) -> None:
        '''
        Setter for attribute `root`
        '''
        self.root = root

    #----------------------------------------

    def get_nodes(self) -> set[MultiScaleComponentTreeNode]:
        '''
        Getter for attribute `nodes`
        '''
        return self.nodes

    def set_nodes(self, nodes: set[MultiScaleComponentTreeNode]) -> None:
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

    #----------------------------------------

    def get_pconverter(self) -> PixelPower2ScaleConverter:
        '''
        Getter for attribure `pconverter`
        '''
        return self.pconverter

    def set_pconverter(self, pconverter: PixelPower2ScaleConverter) -> None:
        '''
        Setter for attribure `pconverter`
        '''
        self.pconverter = pconverter

    #--------------------------------------------------------------------------

    def add_node(self, node: MultiScaleComponentTreeNode) -> None:
        '''
        Adds a node the the MSCT' set of nodes

                Parameters:
                        `node` (MultiScaleComponentTreeNode): a node to add

                Returns:
                        None
        '''
        self.nodes.add(node)

    #--------------------------------------------------------------------------

    def get_node_by_id(self, id: int) -> MultiScaleComponentTreeNode:
        for node in self.get_nodes():
            if node.get_id() == id:
                return node
        return None

    #--------------------------------------------------------------------------

    def init_from_component_tree(self, tree: ComponentTree, pixel_converter: Index2Pixel, scale: int) -> None:
        '''
        Initializes a multiscale component-tree from a component-tree at a lower scale (`n` successive subsampling).

                Parameters:
                        `tree` (ComponentTree) : a component-tree build on a subsampled image
                        `pixel_converter` (Index2Pixel): an index to pixel converter
                        `scale` (int) : the scale at which the pixels are defined

                Returns:
                        None
        '''
        self.set_invert(tree.get_invert())
        indices_to_node = dict()
        to_process = set()
        root = tree.get_root()
        root_node = tree.get_nodes()[root]
        sct_nodes = set()

        self.reset()
        sct_root = MultiScaleComponentTreeNode()
        sct_root.set_level(root_node.get_level())
        sct_root.set_father(sct_root)
        real_pixels = pixel_converter.convert_indices_to_pixels(root_node.get_pixels())
        sct_root.set_pixels_at_scale(real_pixels, scale)
        sct_root.set_area(self.get_pconverter().area_to_global_scale(len(sct_root.get_pixels_at_scale(scale)), scale))
        sct_root.set_subarea(self.get_pconverter().area_to_global_scale(root_node.get_subarea(), scale) - sct_root.get_area())
        self.add_node(sct_root)
        sct_nodes.add(sct_root)
        indices_to_node[root_node.get_index()] = sct_root

        for child in root_node.get_children():
            to_process.add(child)

        while len(to_process) > 0:
            local_node_index = to_process.pop()
            local_node = tree.get_nodes()[local_node_index]

            for child in local_node.get_children():
                to_process.add(child)

            sct_node = MultiScaleComponentTreeNode()
            sct_node.set_level(local_node.get_level())
            father = indices_to_node[local_node.get_father()]
            father.add_child(sct_node)
            sct_node.set_father(father)
            
            real_pixels = pixel_converter.convert_indices_to_pixels(local_node.get_pixels())
            sct_node.set_pixels_at_scale(real_pixels, scale)

            sct_node.set_area(self.get_pconverter().area_to_global_scale(len(sct_node.get_pixels_at_scale(scale)), scale))
            sct_node.set_subarea(self.get_pconverter().area_to_global_scale(local_node.get_subarea(), scale) - sct_node.get_area())
            sct_nodes.add(sct_node)
            indices_to_node[local_node.get_index()] = sct_node

        for sct_node in sct_nodes:
            if sct_node.get_father() is sct_root and sct_node != sct_root:
                sct_root.add_child(sct_node)

        self.set_root(sct_root)
        self.set_nodes(sct_nodes)
        self.compute_all_subareas()

    #--------------------------------------------------------------------------

    def compute_max_tree_height(self) -> int:
        '''
        Computes the max tree height i.e. the height of its deepest branch.

                Parameters:
                        `debug` (bool): whether to print debug informations

                Returns:    
                        `height` (int): the tree max height in [1,255]
        '''
        if self.get_root() is None:
            return 0
        else:
            return self.compute_height(self.get_root(), 0)
        
    #----------------------------------
    
    def compute_height(self, node: MultiScaleComponentTreeNode, height: int) -> int:
        # case 1 : node is a leaf, end of recursive call
        if node.is_leaf():
            return height + 1
        # case 2 : node is not a leaf, recursive call on all children and keep the max value
        else:
            heights = []
            for child in node.get_children():
                branch_height = self.compute_height(child, height + 1)
                heights.append(branch_height)
            return np.max(heights)

    #--------------------------------------------------------------------------

    def compute_all_subareas(self):
        '''
        Computes subareas for all nodes of the MSCT.

                Parameters:
                        None

                Returns:
                        None
        '''
        self.compute_subarea(self.get_root())

    #----------------------------------

    def compute_subarea(self, node: MultiScaleComponentTreeNode) -> int:
        '''
        Computes and sets the subarea attribute for all nodes of the subtree rooted at node.

                Parameters:
                        `node` (MultiScaleComponentTreeNode): the root node

                Returns:
                        None
        '''
        if node.is_leaf():
            node.set_subarea(0)
            return node.get_area()
        else:
            subarea = 0
            for child in node.get_children():
                subarea += self.compute_subarea(child)
            node.set_subarea(subarea)
            return node.get_area() + subarea
        
    #--------------------------------------------------------------------------

    def get_border_pixels(self, node: MultiScaleComponentTreeNode, scale=0) -> tuple[set]:
        '''
        Computes the set of border pixels of a node using its flat zone.

                Parameters:
                        `node` (MultiScaleComponentTreeNode): node representing the flat zone

                Returns:
                        (tuple): set of pixels belonging to the boundary and set of pixels belonging to the interior
        '''
        flat_zone = self.get_flat_zone(node, scale=scale)
        boundary, interior = RCC8.region_to_boundary_and_interior(flat_zone)
        return boundary, interior

    #--------------------------------------------------------------------------

    def compute_mser(self, node: MultiScaleComponentTreeNode, delta: int) -> None:
        '''
        Computes MSER stability values using the given delta.

                Parameters:
                        `node` (MultiScaleComponentTreeNode): MSCT node on which to compute MSER stability
                        `delta` (int): level delta upward and downward in the tree

                Returns:
                        None, assigns `mser`, `area_derivative_delta_level` and `area_derivative_delta_area` attributes of MSCT node
        '''
        for child in node.get_children():
            self.compute_mser(child, delta=delta)

        node.set_mser(maxsize)

        area_node = node.get_total_area()
        area_father = node.get_father().get_total_area()
        level_node = node.get_level()

        iterator = node

        while ((level_node - iterator.get_level()) < delta) and (iterator.get_father() != iterator):
            iterator = iterator.get_father()
            
        if (level_node - iterator.get_level()) >= delta:
            area_father = iterator.get_total_area()

        mser = (area_father - area_node) / area_node
        node.set_mser(mser)

    #--------------------------------------------------------------------------

    def compute_mser_alt(self, node: MultiScaleComponentTreeNode, delta: int) -> None:
        if node is not None:
            for child in node.get_children():
                self.compute_mser_alt(child, delta)

            # upward area
            up_node = self.upward(node, delta, node.get_level())
            up_area = up_node.get_total_area()

            # downward area
            down_nodes = self.downward(node, delta, node.get_level())
            down_areas = 0
            for n in down_nodes:
                down_areas += n.get_total_area()

            mser = (up_area - down_areas) / node.get_total_area()
            node.set_mser(mser)

    #--------------------------------------------------------------------------
    
    def upward(self, node: MultiScaleComponentTreeNode, delta: int, level: int) -> MultiScaleComponentTreeNode:
        if node.get_father() == node:
            return node
        else:
            if level - node.get_level() >= delta:
                return node
            else:
                return self.upward(node.get_father(), delta, level)
            
    #--------------------------------------------------------------------------
            
    def downward(self, node: MultiScaleComponentTreeNode, delta: int, level: int) -> set[MultiScaleComponentTreeNode]:
        if len(node.get_children()) == 0:
            return set([node])
        elif len(node.get_children()) == 1:
            if node.get_level() - level >= delta:
                return set([node])
            else:
                return self.downward(list(node.get_children())[0], delta, level)
        else:
            if node.get_level() - level >= delta:
                return set([node])
            else:
                all_nodes = set()
                for child in node.get_children():
                    cnodes = self.downward(child, delta, level)
                    all_nodes.update(cnodes)
                return all_nodes

    #--------------------------------------------------------------------------

    def filter_by_area(self, area_threshold: int) -> None:
        '''
        Filters the tree using an area criterion and a direct filtering strategy.
        Deactivates the nodes invalidating the criterion.

                Parameters:
                        `area_threshold` (int): min area to keep

                Returns:
                        None
        '''
        for node in self.get_nodes():
            if node.get_area() < area_threshold:
                node.set_active(False)

    #--------------------------------------------------------------------------

    def augment_node_sequential(
        self, node: MultiScaleComponentTreeNode, 
        target_scale: int, scale_image: np.ndarray, 
        pixel_converter: Index2Pixel,
        invert_image: bool
    ) -> MultiScaleComponentTreeNode:
        '''
        Augments a node using a double scale resolution image (intended to be used sequantially).
        Computes a partial component-tree on the support set of node (subtree of node).
        Removes the subtree of node and replaces it with the newly-computed one (at target scale).

                Parameters:
                        `node` (MultiScaleComponentTreeNode): the node to augment
                        `target_scale` (int): scale of the new nodes to be created
                        `scale_image` (ndarray): Numpy array of the image at double scale
                        `pixel_converter` (Index2Pixel): index <-> pixel converter
                        `invert_image` (bool): whether to invert the image

                Returns:
                        (MultiScaleComponentTreeNode): new local root
        '''
        #pc = Index2Pixel(scale_image.shape[1])
        partial_max_trees = self.build_partial_max_trees_from_flatzone(node, scale_image, target_scale, pixel_converter, invert_image)
        self.merge_pmt_with_msct(partial_max_trees, node.get_father(), node)
        return None
    
    #--------------------------------------------------------------------------

    def build_partial_max_trees_from_flatzone(
        self, node: MultiScaleComponentTreeNode, 
        image: np.ndarray, scale: int,
        pixel_converter: Index2Pixel,
        invert_image: bool
    ) -> set[Self]:
        pixels = self.get_local_flat_zone(node, scale)
        mask = self.build_node_content_image(image, pixels)
        ccs, n_ccs = label(mask, structure=[[0,1,0],[1,1,1],[0,1,0]])
        new_cc_indices = []
        partial_max_trees = set()
        for label_index in range(1, n_ccs + 1):
            coords = np.where(ccs == label_index)
            rows = [x for x in coords[0]]
            cols = [x for x in coords[1]]
            cc_pixels = [(rows[x], cols[x]) for x in range(0, len(rows))]
            new_indices = pixel_converter.convert_pixels_to_indices(set(cc_pixels))
            new_cc_indices.append(new_indices)
        for cc_indices in new_cc_indices:
            pmt = MaxTree()
            pmt_mask, pmt_mask_image = self.extract_mask_and_mask_image(image, cc_indices)
            pmt.build_component_tree_from_partial_image(image, pmt_mask, invert_image)
            partial_max_tree = MultiScaleComponentTree()
            partial_max_tree.init_from_component_tree(pmt, pixel_converter, scale)
            partial_max_trees.add(partial_max_tree)
        return partial_max_trees

    #--------------------------------------------------------------------------

    def define_replace_trees(self, partial_max_tree: Self, level: int) -> set[MultiScaleComponentTreeNode]:
        '''
        Defines a tuple composed of two elements : a MSCT and a list of MSCTs.
        The first element is the partial max-tree of be merged/inserted before the old root node's level
        The second element is a list of partial max-trees to replace the subtree rooted in the old root node

                Parameters:
                        `partial_max_tree` (ComponentTree): computed partial max-tree
                        `level` (int): gray-level of the root node upon which the partial max-tree has been computed

                Returns:
                        (tuple): tuple of MSCT, list of MSCT
        '''
        replace_roots = set()
        self.find_replace_roots(partial_max_tree.get_root(), level, replace_roots)
        return replace_roots
    
    #--------------------------------------------------------------------------

    def find_replace_roots(self, node: ComponentTreeNode, level: int, roots: set[ComponentTreeNode]) -> None:
        if node.get_level() > level:
            roots.add(node)
        else:
            for child in node.get_children():
                self.find_replace_roots(child, level, roots)

    #--------------------------------------------------------------------------

    def merge_pmt_with_msct(
        self, 
        partial_max_trees: set[Self],
        insert_node: MultiScaleComponentTreeNode,
        replace_node: MultiScaleComponentTreeNode,
    ) -> None:
        '''
        Merges a partial max-tree at a higher scale with the MSCT at a lower scale.

                Parameters :
                        `partial_max_trees` (set): set of partial max-trees computed from the original node `replace_node`
                        `insert_node` (MultiScaleComponentTreeNode): node at which the replacement subtrees will be inserted (father of `replace_node`)
                        `replace_node` (MultiScaleComponentTreeNode): node on which the partial max-trees are computed as replacement

                Returns : none, updates the MSCT by removing nodes and adding new ones
        '''
        
        pmt_processed = set()

        # step 1 : remove the subtree rooted in replace_node in the MSCT
        self.remove_subtree(replace_node)

        for partial_max_tree in partial_max_trees:

            replace_roots = self.define_replace_trees(partial_max_tree, insert_node.get_level())

            # step 2 : insert a copy of the subtrees of replace_roots (PMT) in the MSCT
            for replace_root in replace_roots:
                nodes_to_insert, copied_ids = self.copy_subtree(replace_root)
                pmt_processed.update(copied_ids.keys())
                self.insert_subtree(insert_node, nodes_to_insert)

            # step 3 : move level per level from the lowest in replace_roots.get_father() to the one of PMT.get_root()
            # either inserting nodes, merging nodes and/or inserting cousin subtrees
            for replace_root in replace_roots:

                upward_branch = self.get_pmt_upward_branch(partial_max_tree, replace_root.get_father())
                msct_node = insert_node

                while (len(upward_branch) > 0):
                    
                    pmt_node = upward_branch.pop(0)
                    # avoiding processing already processed nodes when multiple branches are present
                    while pmt_node.get_id() in pmt_processed and len(upward_branch) > 0:
                        pmt_node = upward_branch.pop(0)

                    cond = True
                    while (cond):

                        # case 1 : pmt.level == msct.level ==> merge to create a multi-scale node
                        if (pmt_node.get_level() == msct_node.get_level()):
                            
                            # merging all lists of pixels on the msct_node
                            for scale in pmt_node.get_scales():
                                to_merge = pmt_node.get_pixels_at_scale(scale)
                                msct_node.merge_pixels_at_scale(to_merge, scale)
                            msct_node.compute_area()
                            # keeping track of the merged node
                            pmt_processed.update(set([pmt_node.get_id()]))
                            # copy added subtrees not existing in the MSCT
                            for child in pmt_node.get_children():
                                if child.get_id() not in pmt_processed and self.decide_subtree_insertion(pmt_processed, child):
                                    nodes_to_insert, copied_ids = self.copy_subtree(child)
                                    pmt_processed.update(copied_ids.keys())
                                    self.insert_subtree(msct_node, nodes_to_insert)
                            # move upward
                            cond = False
                        
                        # case 2 : pmt.level > msct.level ==> insert pmt after msct
                        elif (pmt_node.get_level() > msct_node.get_level()):

                            # copy pmt_node
                            pmt_copy = MultiScaleComponentTreeNode()
                            pmt_copy.set_id(pmt_node.get_id())
                            pmt_copy.set_level(pmt_node.get_level())
                            pmt_copy.set_pixels(pmt_node.get_pixels())
                            pmt_copy.set_area(pmt_node.get_area())
                            pmt_copy.set_subarea(pmt_node.get_subarea())

                            children_to_remove = []
                            for msct_child in msct_node.get_children():
                                if msct_child.get_level() > pmt_node.get_level():
                                    children_to_remove.append(msct_child)
                                    pmt_copy.add_child(msct_child)
                                    msct_child.set_father(pmt_copy)
                            for child_to_remove in children_to_remove:
                                msct_node.remove_child(child_to_remove)

                            msct_node.add_child(pmt_copy)
                            pmt_copy.set_father(msct_node)
                            self.add_node(pmt_copy)
                            pmt_processed.update(set([pmt_node.get_id()]))

                            # connect to every already copied or existing children (exists in inserted_nodes or merged_nodes)
                            # insert subtree if not part of merged or inserted subtrees
                            for child in pmt_node.get_children():
                                if child.get_id() not in pmt_processed and self.decide_subtree_insertion(pmt_processed, child):
                                    nodes_to_insert, copied_ids = self.copy_subtree(child)
                                    pmt_processed.update(copied_ids.keys())
                                    self.insert_subtree(pmt_copy, nodes_to_insert)
                            # move upward
                            cond = False
                        
                        # case 3 : pmt.level < msct.level ==> move upward in msct until msct.level <= pmt.level
                        else:
                            while (msct_node.get_level() > pmt_node.get_level() and msct_node != self.get_root()):
                                msct_node = msct_node.get_father()

                            # stop if reaching root and insert before the root thus creating a new root
                            if (msct_node == self.get_root() and msct_node.get_level() > pmt_node.get_level()):
                                # copy pmt_node
                                pmt_copy = MultiScaleComponentTreeNode()
                                pmt_copy.set_id(pmt_node.get_id())
                                pmt_copy.set_level(pmt_node.get_level())
                                pmt_copy.set_pixels(pmt_node.get_pixels())
                                pmt_copy.set_area(pmt_node.get_area())
                                pmt_copy.set_subarea(pmt_node.get_subarea())
                                pmt_copy.add_child(msct_node)
                                pmt_copy.set_father(pmt_copy)
                                msct_node.set_father(pmt_copy)
                                self.add_node(pmt_copy)
                                pmt_processed.update(set([pmt_node.get_id()]))
                                self.set_root(pmt_copy)
                                msct_node = pmt_copy

                                # connect to every already copied or existing children (exists in inserted_nodes or merged_nodes)
                                # insert subtree if not part of merged or inserted subtrees
                                for child in pmt_node.get_children():
                                    if child not in pmt_processed and self.decide_subtree_insertion(pmt_processed, child):
                                        nodes_to_insert, copied_ids = self.copy_subtree(child)
                                        pmt_processed.update(copied_ids.keys())
                                        self.insert_subtree(pmt_copy, nodes_to_insert)

                                # move upward
                                cond = False
                            else:
                                cond = True

                self.compute_subarea(msct_node)

    #--------------------------------------------------------------------------

    def remove_subtree(self, start: MultiScaleComponentTreeNode) -> None:
        '''
        Removes the subtree rooted in `start` from the MSCT.

                Parameters :
                        `start` (MultiScaleComponentTreeNode): root of the subtree to remove

                Returns : none, removes nodes from self.nodes
        '''
        to_remove = [start]
        while (len(to_remove) > 0):
            node_to_remove = to_remove.pop(0)
            # 1 : remove link to father
            node_to_remove.get_father().remove_child(node_to_remove)
            # 2 : add all children to to_remove list
            for child in node_to_remove.get_children():
                to_remove.append(child)
            # 3 : remove node from MSCT nodes list
            self.get_nodes().remove(node_to_remove)

    #--------------------------------------------------------------------------

    def copy_subtree(self, start: MultiScaleComponentTreeNode) -> tuple[list[MultiScaleComponentTreeNode], dict]:
        '''
        Returns a copy of a subtree.

                Parameters :
                        `start` (MultiScaleComponentTreeNode): root of the subtree to copy

                Returns:
                        `nodes` (list): list of copies of nodes of the subtree rooted in `start`
                        `copied_ids` (dict): dictionnary of mappings from original nodes and their copies
        '''
        nodes = []
        copied_ids = dict()
        to_process = [x for x in start.get_children()]
        # copying the root
        root_copy = MultiScaleComponentTreeNode()
        root_copy.set_id(start.get_id())
        root_copy.set_father(None)
        root_copy.set_pixels(start.get_pixels())
        root_copy.set_area(start.get_area())
        root_copy.set_subarea(start.get_subarea())
        root_copy.set_level(start.get_level())
        nodes.append(root_copy)
        copied_ids[start.get_id()] = root_copy
        # copying all subtree nodes
        while (len(to_process) > 0):
            to_copy = to_process.pop(0)
            node_copy = MultiScaleComponentTreeNode()
            node_copy.set_id(to_copy.get_id())
            father_copy = copied_ids[to_copy.get_father().get_id()]
            node_copy.set_father(father_copy)
            father_copy.add_child(node_copy)
            node_copy.set_pixels(to_copy.get_pixels())
            node_copy.set_area(to_copy.get_area())
            node_copy.set_subarea(to_copy.get_subarea())
            node_copy.set_level(to_copy.get_level())
            nodes.append(node_copy)
            copied_ids[to_copy.get_id()] = node_copy
            for child in to_copy.get_children():
                to_process.append(child)
        return (nodes, copied_ids)
    
    #--------------------------------------------------------------------------
    
    def insert_subtree(
        self, 
        insert_node: MultiScaleComponentTreeNode, 
        copied_nodes: set[MultiScaleComponentTreeNode]
    ) -> None:
        '''
        Inserts a subtree at the given node in the MSCT.

                Parameters :
                        `insert_node` (MultiScaleComponentTreeNode): node of the MSCT after which the subtree will be inserted
                        `copied_nodes` (set): set of nodes to add to the MSCT

                Returns : none, inserts new nodes into self.nodes
        '''
        # adding all copied nodes to the list of nodes of the MSCT
        for node in copied_nodes:
            self.add_node(node)
        # adding the link from the root of the copied subtree to the insert_node
        insert_node.add_child(copied_nodes[0])
        copied_nodes[0].set_father(insert_node)

    #--------------------------------------------------------------------------

    def get_pmt_upward_branch(
        self, 
        partial_max_tree: Self, 
        start: MultiScaleComponentTreeNode
    ) -> list[MultiScaleComponentTreeNode]:
        '''
        Returns a list of all nodes from the given root to the root of the local tree.

                Parameters :
                        `partial_max_tree` (MultiScaleComponentTreeNode): partial max-tree on which the branch is extracted
                        `start` (MultiScaleComponentTreeNode): root node at which the branch starts

                Returns :
                        `nodes` (list): list of ancestor nodes from `start` to the root of `partial_max_tree`
        '''
        nodes = []
        node = start
        father = node.get_father()
        while (node != father):
            nodes.append(node)
            node = father
            father = node.get_father()
        nodes.append(partial_max_tree.get_root())
        return nodes
    
    #--------------------------------------------------------------------------
    
    def get_subtree_ids(self, root: MultiScaleComponentTreeNode) -> set[int]:
        '''
        Returns a set of node IDs for all nodes belonging to the subtree rooted in `start`

                Parameters :
                        `root` (MultiScaleComponentTreeNode): root node from which to start gathering IDs

                Returns :
                        `ids` (set): set of node IDs for the subtree rooted in `root`
        '''
        ids = set()
        to_process = [root]
        while len(to_process) > 0:
            node = to_process.pop(0)
            ids.add(node.get_id())
            for child in node.get_children():
                to_process.append(child)
        return ids
    
    #--------------------------------------------------------------------------
    
    def decide_subtree_insertion(self, processed_ids: set[int], root: MultiScaleComponentTreeNode) -> bool:
        '''
        Decices whether the partial subtree rooted in `root` has to be copied and inserted in the MSCT

                Parameters :
                        `processed_ids` (set): set of node IDs of already inserted/merged partial max-tree nodes in the MSCT
                        `root` (MultiScaleComponentTreeNode): root of the partial max-tree subtree to insert

                Returns :
                        `insert` (bool): True if the subtree is to be inserted and False otherwise
        '''
        # insertion if none of the subtree nodes alreay exists in the MSCT
        subtree_ids = self.get_subtree_ids(root)
        intersection = processed_ids.intersection(subtree_ids)
        if len(intersection) == 0:
            return True
        else:
            return False

    #--------------------------------------------------------------------------

    def build_node_content_image(self, image: np.ndarray, pixels: set[tuple[int, int]]) -> np.ndarray:
        '''
        Returns an image with the given pixels as foreground on a black background.

                Parameters :
                        `image` (ndarray): reference image
                        `pixels` (set): set of pixels

                Returns :
                        `zone` (ndarray): image with `pixels` as foreground
        '''
        zone = np.zeros(image.shape, dtype=image.dtype)
        for x, y in pixels:
            zone[x, y] = 255
        return zone

    #--------------------------------------------------------------------------

    def gather_subtree_nodes(self, node: MultiScaleComponentTreeNode) -> set[MultiScaleComponentTreeNode]:
        '''
        Returns a set containing all nodes of the subtree of node.

                Parameters:
                        `node` (MultiScaleComponentTreeNode): the root node

                Returns:
                        processed (set): a set containing all nodes in the subtree of node
        '''
        to_process = set()
        processed = set()
        to_process.add(node)
        while len(to_process) > 0:
            n = to_process.pop()
            for child in n.get_children():
                to_process.add(child)
            processed.add(n)
        return processed

    #--------------------------------------------------------------------------

    def extract_mask_and_mask_image(self, scale_image: np.ndarray, indices: set[int]) -> np.ndarray:
        '''
        Returns a binary mask and a grayscale image of a flatzone.

                Parameters :
                        `scale_image` (ndarray): reference image
                        `indices` (set): set of pixel indices (flatten in 1D)
                
                Returns :
                        `mask` (ndarray): binary mask of the flatzone
                        `mask_image` (ndarray): grayscale image of the flatzone
        '''
        mask = np.zeros(scale_image.shape, dtype=bool)
        mask_image = np.zeros(scale_image.shape, dtype=int)
        flat_scale = scale_image.flatten()
        mask = mask.flatten()
        mask_image = mask_image.flatten()
        for index in indices:
            mask[index] = True
            mask_image[index] = flat_scale[index]
        return mask, mask_image

    #--------------------------------------------------------------------------

    def get_local_flat_zone(self, node: MultiScaleComponentTreeNode, scale: int) -> set[tuple[int, int]]:
        '''
        Returns all pixels of the flat zone represented by the given node at the same scale.

                Parameters:
                        `node` (MultiScaleComponentTreeNode): the root node
                        `scale` (int): target scale to express the flat zone

                Returns:
                        pixels (set): set of pixels forming the flat zone of node
        '''
        pc = PixelPower2ScaleConverter()
        pixels = set()
        to_process = set()
        to_process.add(node)

        while len(to_process) > 0:
            node = to_process.pop()
            for node_scale in node.get_scales():
                node_pixels = node.get_pixels_at_scale(node_scale)
                node_pixels_upscaled = pc.convert_pixels_to_n_upper_scale(node_pixels, node_scale - scale)
                pixels.update(node_pixels_upscaled)
            for child in node.get_children():
                to_process.add(child)
        return pixels
    
    #--------------------------------------------------------------------------
    
    def get_node_pixels(self, node: MultiScaleComponentTreeNode, scale: int) -> set:
        '''
        Returns all pixels stored in the given node at the given scale.

                Parameters:
                        `node` (MultiScaleComponentTreeNode): the root node
                        `scale` (int): target scale to express the flat zone

                Returns:
                        pixels (set): set of pixels forming the flat zone of node
        '''
        pc = PixelPower2ScaleConverter()
        pixels = set()

        for node_scale in node.get_scales():
            node_pixels = node.get_pixels_at_scale(node_scale)
            node_pixels_upscaled = pc.convert_pixels_to_n_upper_scale(node_pixels, node_scale - scale)
        pixels.update(node_pixels_upscaled)

        return pixels

    #--------------------------------------------------------------------------

    def build_channel_histogram(self, channel_images: list[np.ndarray]) -> None:
        '''
        Builds a histogram composed of all channel images data for each MSCT node

                Parameters :
                        `channel_images` (list): list of channel images except the base channel used to build the MSCT

                Returns : none
        '''
        self.rec_build_channel_histogram(self.get_root(), channel_images)

    #--------------------------------------------------------------------------

    def rec_build_channel_histogram(self, node: MultiScaleComponentTreeNode, channel_images: list[np.ndarray]) -> list[int]:
        '''
        Recursively builds a channel histogram an returns the subtree's histogram

                Parameters :
                        `node` (MultiScaleComponentTreeNode): node upon which the histogram is computed
                        `channel_images` (list): list of channel images except the base channel image

                Returns :
                        `hist` (list): locally computed histogram for `node`
        '''
        # local histogram
        hist = [0 for i in range(0, len(channel_images))]
        fz = self.get_node_content(node)
        for index in range(0, len(channel_images)):
            channel_image = channel_images[index]
            value = 0
            for pixel in fz:
                value += channel_image[pixel]
            hist[index] = value

        # children histograms
        for child in node.get_children():
            child_hist = self.rec_build_channel_histogram(child, channel_images)
            for i in range(0, len(child_hist)):
                hist[i] += child_hist[i]

        # complete histogram
        node.set_histogram(hist)
        return hist

    #--------------------------------------------------------------------------

    def save_dot(self, filename: str, mode='dot') -> None:
        '''
        Saves the MSCT in the DOT language in a dot file usinh and PyDot/GraphViz

                Parameters:
                        `filename` (str): path to the output file without extension
                        `mode` (str): 'png' or 'dot' or 'svg' whether to save the tree as a PNG file, a DOT file or a SVG file

                Returns:
                        None
        '''
        graph = pydot.Dot('multi_scale_component_tree', graph_type='graph', bgcolor='white')

        # add all nodes from root to leaves
        for node in self.get_nodes():
            id_level = f"{node.get_id()} ({node.get_level()})"
            areas = f"area {node.get_area()}/{node.get_subarea()}/{node.get_total_area()}"
            mser = f"{node.get_mser()}"
            pixel_dict = ""
            for scale in node.get_scales():
                pixel_dict += f" {scale}:{len(node.get_pixels_at_scale(scale))}"
            histogram = " "
            for i in range(0, len(node.get_histogram())):
                histogram += f"{node.get_histogram()[i]} "
            label_text = f"{id_level}\n{areas}\n{mser}\n{pixel_dict}\n{histogram}"

            node = pydot.Node(
                name=f"node_{node.get_id()}", label=label_text)
            graph.add_node(node)

        # connect nodes
        process = set()
        process.add(self.get_root())

        while len(process) > 0:
            cur_node = process.pop()
            for child in cur_node.get_children():
                process.add(child)
                edge = pydot.Edge(f"node_{cur_node.get_id()}", f"node_{child.get_id()}")
                graph.add_edge(edge)

        if mode == 'png':
            graph.write_png(f"{filename}.png")
        elif mode == 'dot':
            graph.write_dot(f"{filename}.dot")
        elif mode == 'svg':
            graph.write_svg(f"{filename}.svg")

    #--------------------------------------------------------------------------

    def save_dot_highlight(self, filename: str, nodes: set, mode='dot') -> None:
        '''
        Saves the MSCT in the DOT language in a dot file usinh and PyDot/GraphViz

                Parameters:
                        `filename` (str): path to the output file without extension
                        `nodes` (set): set of nodes to highlight
                        `mode` (str): 'png' or 'dot' or 'svg' whether to save the tree as a PNG file, a DOT file or a SVG file

                Returns:
                        None
        '''
        graph = pydot.Dot('multi_scale_component_tree', graph_type='graph', bgcolor='white')

        # add all nodes from root to leaves
        for node in self.get_nodes():
            id_level = f"{node.get_id()} ({node.get_level()})"
            areas = f"area {node.get_area()}/{node.get_subarea()}/{node.get_total_area()}"
            mser = f"{node.get_mser()}"
            pixel_dict = ""
            for scale in node.get_scales():
                pixel_dict += f" {scale}:{len(node.get_pixels_at_scale(scale))}"
            histogram = " "
            for i in range(0, len(node.get_histogram())):
                histogram += f"{node.get_histogram()[i]} "
            label_text = f"{id_level}\n{areas}\n{mser}\n{pixel_dict}\n{histogram}"
            if node in nodes:
                gnode = pydot.Node(name=f"node_{node.get_id()}", label=label_text, style='filled', fillcolor='#40e0d0')
                graph.add_node(gnode)
            else:
                gnode = pydot.Node(name=f"node_{node.get_id()}", label=label_text)
                graph.add_node(gnode)

        # connect nodes
        process = set()
        process.add(self.get_root())

        while len(process) > 0:
            cur_node = process.pop()
            for child in cur_node.get_children():
                process.add(child)
                edge = pydot.Edge(f"node_{cur_node.get_id()}", f"node_{child.get_id()}")
                graph.add_edge(edge)

        
        if mode == 'png':
            graph.write_png(f"{filename}.png")
        elif mode == 'dot':
            graph.write_dot(f"{filename}.dot")
        elif mode == 'svg':
            graph.write_svg(f"{filename}.svg")

    #--------------------------------------------------------------------------

    def save_dot_from_node(self, filename: str, node: MultiScaleComponentTreeNode, mode='dot') -> None:
        '''
        Saves the MSCT in the DOT language in a dot file usinh and PyDot/GraphViz

                Parameters:
                        `filename` (str): path to the output file without extension
                        `node` (MultiScaleComponentTreeNode): root node
                        `mode` (str): 'png' or 'dot' or 'svg' whether to save the tree as a PNG file, a DOT file or a SVG file

                Returns:
                        None
        '''
        graph = pydot.Dot('multi_scale_component_tree', graph_type='graph', bgcolor='white')

        # add all nodes from node to its subtree leaves
        process = set()
        process.add(node)

        while len(process) > 0:
            cur_node = process.pop()
            id_level = f"{cur_node.get_id()} ({cur_node.get_level()})"
            areas = f"area {cur_node.get_area()}/{cur_node.get_subarea()}/{cur_node.get_total_area()}"
            mser = f"{cur_node.get_mser()}"
            pixel_dict = ""
            for scale in cur_node.get_scales():
                pixel_dict += f" {scale}:{len(cur_node.get_pixels_at_scale(scale))}"
            histogram = " "
            for i in range(0, len(cur_node.get_histogram())):
                histogram += f"{cur_node.get_histogram()[i]} "
            label_text = f"{id_level}\n{areas}\n{mser}\n{pixel_dict}\n{histogram}"
            gnode = pydot.Node(name=f"node_{cur_node.get_id()}", label=label_text)
            graph.add_node(gnode)
            
            for child in cur_node.get_children():
                process.add(child)

        # connect nodes
        process = set()
        process.add(node)

        while len(process) > 0:
            cur_node = process.pop()
            for child in cur_node.get_children():
                process.add(child)
                edge = pydot.Edge(f"node_{cur_node.get_id()}", f"node_{child.get_id()}")
                graph.add_edge(edge)

        if mode == 'png':
            graph.write_png(f"{filename}.png")
        elif mode == 'dot':
            graph.write_dot(f"{filename}.dot")
        elif mode == 'svg':
            graph.write_svg(f"{filename}.svg")

    #--------------------------------------------------------------------------

    def save_dot_highlight_from_node(self, filename: str, node: MultiScaleComponentTreeNode, nodes: set, mode='dot') -> None:
        '''
        Saves the MSCT in the DOT language in a dot file usinh and PyDot/GraphViz

                Parameters:
                        `filename` (str): path to the output file without extension
                        `node` (MultiScaleComponentTreeNode): root
                        `nodes` (set): set of nodes to highlight
                        `mode` (str): 'png' or 'dot' or 'both' whether to save the tree as a PNG file, a DOT file or a SVG file

                Returns:
                        None
        '''
        graph = pydot.Dot('multi_scale_component_tree', graph_type='graph', bgcolor='white')

        # add all nodes from node to its subtree leaves
        process = set()
        process.add(node)

        while len(process) > 0:
            cur_node = process.pop()
            id_level = f"{cur_node.get_id()} ({cur_node.get_level()})"
            areas = f"area {cur_node.get_area()}/{cur_node.get_subarea()}/{cur_node.get_total_area()}"
            mser = f"{cur_node.get_mser()}"
            pixel_dict = ""
            for scale in cur_node.get_scales():
                pixel_dict += f" {scale}:{len(cur_node.get_pixels_at_scale(scale))}"
            histogram = " "
            for i in range(0, len(cur_node.get_histogram())):
                histogram += f"{cur_node.get_histogram()[i]} "
            label_text = f"{id_level}\n{areas}\n{mser}\n{pixel_dict}\n{histogram}"
            if cur_node in nodes:
                gnode = pydot.Node(name=f"node_{cur_node.get_id()}", label=label_text, style='filled', fillcolor='#40e0d0')
                graph.add_node(gnode)
            else:
                gnode = pydot.Node(name=f"node_{cur_node.get_id()}", label=label_text)
                graph.add_node(gnode)

            for child in cur_node.get_children():
                process.add(child)

        # connect nodes
        process = set()
        process.add(node)

        while len(process) > 0:
            cur_node = process.pop()
            for child in cur_node.get_children():
                process.add(child)
                edge = pydot.Edge(f"node_{cur_node.get_id()}", f"node_{child.get_id()}")
                graph.add_edge(edge)

        if mode == 'png':
            graph.write_png(f"{filename}.png")
        elif mode == 'dot':
            graph.write_dot(f"{filename}.dot")
        elif mode == 'svg':
            graph.write_svg(f"{filename}.svg")

    #--------------------------------------------------------------------------

    def save_dot_python(self, filename: str) -> None:
        '''
        Same as save_dot but without using PyDot/GraphViz and only with simple ids

                Parameters:
                        `filename` (str): path to the output file without extension

                Returns:
                        None
        '''
        filepath = f"{filename}.dot"
        with open(filepath, 'w') as outfile:

            outfile.write("digraph {\n")

            for node in self.get_nodes():
                if len(node.get_children()) > 0:
                    for child in node.get_children():
                        outfile.write(f"\t{node.get_id()} -> {child.get_id()};\n")

            outfile.write("}\n")

    #--------------------------------------------------------------------------

    def reconstruct_subtree(self, root: MultiScaleComponentTreeNode, image: np.ndarray) -> np.ndarray:
        '''
        Reconstructs a subtree rooted at a given node, 
        Creates an image with the background in black (0) and pixels having their true gray-level value from the MSCT.

                Parameters:
                        `node` (MultiScaleComponentTreeNode): node for which to construct the flat zone
                        `image` (ndarray): Numpy array of the original image

                Returns:
                        reconstructed (ndarray): reconstructed grayscale subtree image
        '''    
        subtree_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        to_process = []
        to_process.append(root)
        while len(to_process) > 0:
            node = to_process.pop(0)
            pixels = self.get_node_content(node)
            level = node.get_level()
            for row, col in pixels:
                subtree_img[row, col] = level
            for child in node.get_children():
                to_process.append(child)
        return subtree_img
    
    #--------------------------------------------------------------------------

    def reconstruct_flat_zone(self, node: MultiScaleComponentTreeNode, image: np.ndarray, binary=False) -> np.ndarray:
        '''
        Reconstructs a flat zone at a given node, if binary is true, creates a binary image
        with the flat zone in white (255) and the background in black (0)
        else creates an image with the background in black (0) and the flat zone in the gray level
        of its node.

                Parameters:
                        `node` (MultiScaleComponentTreeNode): node for which to construct the flat zone
                        `image` (ndarray): Numpy array of the original image
                        `binary` (bool): if True the uses 255 for the FZ, otherwise use the node's (gray) level

                Returns:
                        reconstructed (ndarray): reconstructed grayscale flat zone image
        '''
        reconstructed = np.zeros(image.shape, dtype=np.uint8)
        to_process = [node]
        while len(to_process) > 0:
            n = to_process.pop(0)
            for child in n.get_children():
                to_process.append(child)
            pixels = self.get_node_content(n)
            if binary:
                level = 255
            else:
                level = n.get_level()
            for pixel in pixels:
                reconstructed[pixel] = level
        return reconstructed
    
    #--------------------------------------------------------------------------

    def get_node_content(self, node: MultiScaleComponentTreeNode) -> set:
        '''
        Returns all pixels of the node at the top scale

                Parameters:
                        `root` (MultiScaleComponentTreeNode): node for which the flat zone has to be fetched

                Returns:
                        all_pixels (set): set of pixels of the flat zone
        '''
        all_pixels = set()
        for scale in node.get_scales():
            pixels = node.get_pixels_at_scale(scale)
            upscaled_pixels = self.get_pconverter().convert_pixels_to_n_upper_scale(pixels, scale)
            all_pixels.update(upscaled_pixels)
        return all_pixels

    #--------------------------------------------------------------------------

    def reconstruct_image(self, image: np.ndarray) -> np.ndarray:
        '''
        Reconstructs an image using the MSCT.

                Parameters:
                        `image` (ndarray): Numpy array of the original image

                Returns:
                        reconstructed (ndarray): Numpy array of the reconstructed image
        '''
        return self.reconstruct_flat_zone(self.get_root(), image, binary=False)
    
    #--------------------------------------------------------------------------

    def get_active_ascendant_level(self, node: MultiScaleComponentTreeNode) -> int:
        '''
        Returns the closest ascendent level that is active, considering node inactive

                Parameters:
                        `node` (MultiScaleComponentTreeNode): the current node to process

                Returns:
                        (int) the gray level of the node being the closest ascendent which is active
        '''
        if node.father() is None:
            return self.get_root().get_level()
        else:
            if node.get_father().get_active():
                return node.get_father().get_level()
            else:
                return self.get_active_ascendant_level(node.get_father())
            
    #--------------------------------------------------------------------------

    def get_total_pixels_stored(self) -> int:
        '''
        Returns the total number of pixels stored at their respective scales
        '''
        nb_pixels = 0
        to_process = []
        to_process.append(self.get_root())
        while len(to_process) > 0:
            local_node = to_process.pop(0)
            for scale in local_node.get_scales():
                nb_pixels += len(local_node.get_pixels_at_scale(scale))
            for child in local_node.get_children():
                to_process.append(child)
        return nb_pixels

    #--------------------------------------------------------------------------

    def export_tree_as_csv(self, output_dir: str, output_name: str):
        '''
        Exports the tree nodes as lines in a CSV file.

                Parameters:
                        `output_dir` (str): output directory to save the CSV file
                        `output_name` (str): CSV filename without extension

                Returns:
                        None
        '''
        node_attributes = []
        for node in self.get_nodes():
            node_attributes.append(node.get_attributes_dict())

        fields = node_attributes[0].keys()
        filepath = join(output_dir, f"{output_name}.csv")

        with open(filepath, 'w', newline='') as csv_file:
            writer = DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            for attributes in node_attributes:
                writer.writerow(attributes)
