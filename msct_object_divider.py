#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import os
import numpy as np
import matplotlib.pyplot as plt

from otsu import Otsu
from shaping import ShapingTreeNode, MSCTShapingTree, MSCTShapingTreeNode, Shaping
from multi_scale_component_tree import MultiScaleComponentTree
from multi_scale_component_tree_node import MultiScaleComponentTreeNode
from mser import MSER
from mser_tree import MSERNode, MSERTree
from msct_utils import save_image
from skimage.transform import resize
from scipy.ndimage import binary_erosion, label
from scipy.signal import convolve2d
from sklearn.cluster import KMeans
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes

from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
from distinctipy import distinctipy

#------------------------------------------------------------------------------

class MSCTObjectDivider:
     
    def init(self):
        pass

    #--------------------------------------------------------------------------

    @staticmethod
    def fit_ellipse_to_shape(width: int, height: int, contour_pixels: set, interior_pixels: set, name: str):
        '''
        Fits an ellipse to the shape formed by a flat zone using the least squares method.

                Parameters:
                        `width` (int): width of the image (higest scale)
                        `height` (int): height of the image (highest scale)
                        `contour_pixels` (set): set of pixels of the border of the node's flat zone's shape
                        `name` (str): name for saving the superimposed ellipse on the shape

                Returns:
                        `image_shape` (ndarray): binary image of the flat zone's shape
                        `image_ellipse` (ndarray): binary image of the unfilled best fit ellipse
        '''

        # set to column vectors
        X = np.zeros((len(contour_pixels), 1))
        Y = np.zeros((len(contour_pixels), 1))
        i = 0
        for row, col in contour_pixels:
            X[i,0] = col
            Y[i,0] = height - 1 - row
            i += 1

        # test with scikit-image EllipseModel
        points = np.array(list(contour_pixels))
        pts_x = points[:, 0]
        pts_y = points[:, 1]

        ell = EllipseModel()
        ell.estimate(points)

        xc, yc, a, b, theta = ell.params

        '''fig, axs = plt.subplots(1, 1)
        axs.scatter(pts_x, pts_y)
        axs.scatter(xc, yc, color='red', s=100)
        axs.set_xlim(0, width)
        axs.set_ylim(0, height)'''

        ell_patch = Ellipse((xc, yc), 2*a, 2*b, theta*180/np.pi, edgecolor='red', facecolor='None')
        image_ellipse = np.zeros((height * width), dtype=np.uint8)
        test_pts = np.zeros((height * width, 2))
        for line in range(0, height):
            test_pts[line*width:(line+1)*width] = np.array([(line, i) for i in range(0, width)])
        validity = ell_patch.contains_points(test_pts)
        image_ellipse[np.where(validity == True)] = 255
        image_ellipse = image_ellipse.reshape((height, width))

        '''axs.add_patch(ell_patch)
        plt.savefig(os.path.join('output', f"TEST_{name}.png"))
        plt.delaxes(axs)
        plt.close()'''

        image_shape = np.zeros((height, width), dtype=np.uint8)
        for row, col in contour_pixels:
            image_shape[row, col] = 1
        for row, col in interior_pixels:
            image_shape[row, col] = 1

        '''save_image(image_shape, f"shape_{name}", 'output')
        save_image(image_ellipse, f"ellipse_{name}", 'output')'''

        return image_shape, image_ellipse

    #--------------------------------------------------------------------------

    @staticmethod
    def fit_ellipse_to_shape_scaled(candidate, msct: MultiScaleComponentTree, width: int, height: int, name: str):
        boundary, interior = msct.get_border_pixels(candidate)
        mat = np.zeros((height, width), dtype=np.uint8)
        h, w = list(boundary)[0]
        lines = []
        columns = []
        for l, c in boundary:
            mat[l,c] = 1
            lines.append(l)
            columns.append(c)
        for l, c in interior:
            mat[l,c] = 1
            lines.append(l)
            columns.append(c)
        h_min = np.min(lines)
        h_max = np.max(lines)
        w_min = np.min(columns)
        w_max = np.max(columns)
        size_h = h_max - h_min + 1
        size_w = w_max - w_min + 1
        size = max(size_h, size_w)
        shape = mat[h_min:h_max+1, w_min:w_max+1]

        preserved_ar_shape = np.zeros((size, size), np.uint8)
        start_h = 0
        start_w = 0
        if size_h % 2 == 0:
            if size % 2 == 0:
                start_h = (size // 2) - (size_h // 2)
            else:
                start_h = (size // 2 + 1) - (size_h // 2)
        else:
            if size % 2 == 0:
                start_h = (size // 2) - (size_h // 2 + 1)
            else:
                start_h = (size // 2 + 1) - (size_h // 2 + 1)

        if size_w % 2 == 0:
            if size % 2 == 0:
                start_w = (size // 2) - (size_w // 2)
            else:
                start_w = (size // 2 + 1) - (size_w // 2)
        else:
            if size % 2 == 0:
                start_w = (size // 2) - (size_w // 2 + 1)
            else:
                start_w = (size // 2 + 1) - (size_w // 2 + 1)

        preserved_ar_shape[start_h : start_h + size_h, start_w : start_w + size_w] = shape
        scaled_shape = resize(preserved_ar_shape, (100, 100), preserve_range=True, anti_aliasing=False, mode='constant', order=0)

        import cv2
        contours = cv2.findContours(np.array(scaled_shape*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        cntr = contours[0]
        contour_image = np.zeros((100, 100), dtype=np.uint8)
        crows = []
        ccols = []
        for pixel in cntr:
            l = pixel[0,1]
            c = pixel[0,0]
            contour_image[l, c] = 1
            crows.append(l)
            ccols.append(c)
        #save_image(contour_image*255, f"{name}_contour", 'output')

        # set to column vectors
        scaled_x = []
        scaled_y = []
        points = []
        for row in range(0, contour_image.shape[0]):
            for col in range(0, contour_image.shape[1]):
                if contour_image[row,col] != 0:
                    scaled_x = col
                    scaled_y = contour_image.shape[0] - 1 - row
                    points.append((row, col))
        scaled_x = np.array(scaled_x)
        scaled_y = np.array(scaled_y)
        points = np.array(points)

        # test with scikit-image EllipseModel
        pts_x = points[:, 0]
        pts_y = points[:, 1]

        ell = EllipseModel()
        ell.estimate(points)

        xc, yc, a, b, theta = ell.params

        '''fig, axs = plt.subplots(1, 1)
        axs.scatter(pts_x, pts_y)
        axs.scatter(xc, yc, color='red', s=100)
        axs.set_xlim(0, width)
        axs.set_ylim(0, height)'''

        ell_patch = Ellipse((xc, yc), 2*a, 2*b, theta*180/np.pi, edgecolor='red', facecolor='None')
        image_ellipse = np.zeros((100 * 100), dtype=np.uint8)
        test_pts = np.zeros((100 * 100, 2))
        for line in range(0, 100):
            test_pts[line*100:(line+1)*100] = np.array([(line, i) for i in range(0, 100)])
        validity = ell_patch.contains_points(test_pts)
        image_ellipse[np.where(validity == True)] = 1
        image_ellipse = image_ellipse.reshape((100, 100))

        contour_row_min = min(crows)
        contour_row_max = max(crows)
        contour_col_min = min(ccols)
        contour_col_max = max(ccols)
        image_ellipse[:contour_row_min,:] = 0
        image_ellipse[contour_row_max+1:,:] = 0
        image_ellipse[:,:contour_col_min] = 0
        image_ellipse[:,contour_col_max+1:] = 0

        '''axs.add_patch(ell_patch)
        plt.savefig(os.path.join('output', f"TEST_fortesr.png"))
        plt.delaxes(axs)
        plt.close()'''

        #save_image(scaled_shape*255, f"{name}_shape", 'output')
        #save_image(image_ellipse*255, f"{name}_ellipse", 'output')

        return scaled_shape, image_ellipse
    
    #--------------------------------------------------------------------------
    
    @staticmethod
    def compute_mse_images(image_a: np.ndarray, image_b: np.ndarray) -> float:
        '''
        Computes the mean square error (MSE) of two equal size binary images.

                Parameters:
                        `image_a` (ndarray): first binary image
                        `image_b` (ndarray): second binary image

                Returns:
                        mse (float): mean square error between `image_a` and `image_b`
        '''
        assert image_a.shape[0] == image_b.shape[0] and image_a.shape[1] == image_b.shape[1]
        mse = np.sum(np.power(image_a.astype(np.uint8) - image_b.astype(np.uint8), 2))
        mse /= (image_a.shape[0] * image_a.shape[1])
        return mse
    
    #--------------------------------------------------------------------------
    
    @staticmethod
    def compute_all_ellipses(candidates: list, msct: MultiScaleComponentTree, width: int, height: int, image: np.ndarray) -> set[MultiScaleComponentTreeNode]:
        all_objects = set()
        for candidate in candidates:
            to_process = []
            to_process.append(candidate)
            while len(to_process) > 0:
                node = to_process.pop(0)
                mat_shape, mat_ellipse = MSCTObjectDivider.fit_ellipse_to_shape_scaled(node, msct, width, height, f"{candidate.get_id()}_{node.get_id()}")
                mse = MSCTObjectDivider.compute_mse_images(mat_shape, mat_ellipse)
                node.fellipse = mse
                for child in node.get_children():
                    to_process.append(child)
            msct.save_dot_from_node(os.path.join('output', f"msct_{candidate.get_id()}"), candidate)
            mser_tree = MSCTObjectDivider.compute_mser_tree(candidate)
            objects = MSCTObjectDivider.subdivide_mser(mser_tree, max_mser=1.5)
            # extract original cluster
            fz = msct.get_local_flat_zone(candidate, 0)
            fz_img = msct.build_node_content_image(image, fz)
            save_image(fz_img, f"cluster_{candidate.get_id()}", 'output')
            # extract individual cells
            for obj in objects:
                fz_obj = msct.get_local_flat_zone(obj, 0)
                fz_obj_img = msct.build_node_content_image(image, fz_obj)
                save_image(fz_obj_img, f"cell_{candidate.get_id()}_{obj.get_id()}", 'output')
            all_objects.update(objects)
        return all_objects
    
    #--------------------------------------------------------------------------
    
    @staticmethod
    def subdivide_objects_ellipse(
        candidate, msct: MultiScaleComponentTree, 
        width: int, height: int, 
        mse: float, mse_factor: float, area_factor: float
    ) -> list:
        '''
        Subdivides a cluster to identify individual objects based on its local max-tree.

                Parameters:
                        `candidate` (MultiScaleComponentTreeNode): root node of the local cluster to subdivide
                        `msct` (MultiScaleComponentTree): global MSCT to which `candidate` belongs to
                        `width` (int): image width (highest scale)
                        `height` (int): image height (highest scale)
                        `mse` (float): Mean Square Error of `candidate` and its best fit ellipse 
                        `mse_factor` (float): reduction factor of `mse` to deem an object valid [0,1] (max allowed value) 
                        `area_factor` (float): reduction factor of `candidate`'s area to deem an object valid [0,1] (min allowed value)

                Returns:
                        `objects` (list): list of nodes whose flat zones represents objets of the cluster rooted in `candidate`
        '''
        #min_area = max(16, candidate.get_total_area() * area_factor)
        min_area = 16
        print(f"SUBDIVIDE {candidate.get_id()} (min_area {min_area})")
        objects = MSCTObjectDivider.subdivide_objects_ellipse_rec(
            candidate=candidate, msct=msct, 
            width=width, height=height,
            mse=mse, mse_factor=mse_factor, min_area=min_area
        )
        return objects
    
    #--------------------------------------------------------------------------

    @staticmethod
    def subdivide_objects_ellipse_rec(
        candidate, msct: MultiScaleComponentTree, 
        width: int, height: int,
        mse: float, mse_factor: float, min_area: int
    ) -> list:
        '''
        Recursively subdivides nodes until stabilization (no more candidate to process).

                Parameters:
                        `candidate` (MultiScaleComponentTreeNode): root node of the local cluster to subdivide
                        `msct` (MultiScaleComponentTree): global MSCT to which `candidate` belongs to
                        `width` (int): image width (highest scale)
                        `height` (int): image height (highest scale)
                        `mse` (float): Mean Square Error of `candidate` and its best fit ellipse 
                        `mse_factor` (float): reduction factor of `mse` to deem an object valid [0,1] (max allowed value) 
                        `area_factor` (float): reduction factor of `candidate`'s area to deem an object valid [0,1] (min allowed value)

                Returns:
                        `objects` (list): list of nodes whose flat zones represents objets of the cluster rooted in `candidate`
        '''
        max_mse = mse * mse_factor
        # 16 pixels = 1 when using 2 scales
        #min_area = max(16, candidate.get_total_area() * area_factor)
        #min_area = candidate.get_total_area() * area_factor

        # go down along the branch until a node with mode than one child is found
        node = candidate
        while len(node.get_children()) > 0 and len(node.get_children()) < 2:
            node = list(node.get_children())[0]
            
        objects = []
        if len(node.get_children()) > 1:
            # -> for each child
            for child in node.get_children():

                # -> case 1 : if MSE of child is at most mse * mse_factor
                print(f"\tarea decision : {child.get_total_area()} ? {min_area}")
                if child.get_total_area() >= min_area:

                    # computing MSE
                    boundary, interior = msct.get_border_pixels(child)
                    mat_shape, mat_ellipse = MSCTObjectDivider.fit_ellipse_to_shape(width, height, boundary, interior, f"subplot_{child.get_id()}")
                    child_mse = MSCTObjectDivider.compute_mse_images(mat_shape, mat_ellipse)
                    print(f"\tchild {child.get_id()} with MSE {child_mse} (max is {max_mse})")

                    if child_mse <= max_mse:
                        sub_objects = MSCTObjectDivider.subdivide_objects_rec(
                            candidate=child, msct=msct, 
                            width=width, height=height,
                            min_area=min_area, mse=mse, mse_factor=mse_factor
                        )
                        if len(sub_objects) > 1:
                            for subo in sub_objects:
                                objects.append(subo)
                        else:
                            objects.append(child)                           
                    else:
                        if child_mse <= 1 - ((1 - mse_factor) / 2):
                            # -> -> process further with recursive call on child and add sub objects to objects
                            sub_objects = MSCTObjectDivider.subdivide_objects_rec(
                                candidate=child, msct=msct, 
                                width=width, height=height,
                                min_area=min_area, mse=mse, mse_factor=mse_factor
                            )
                            for subo in sub_objects:
                                objects.append(subo)

        # if less than 2 objects then candidate is a better fit, otherwise returns objects
        if len(objects) < 2:
            return [candidate]
        else:
            return objects
    
    #--------------------------------------------------------------------------

    @staticmethod
    def compute_object_subdivision(
        candidate: MultiScaleComponentTreeNode, 
        msct: MultiScaleComponentTree, 
        image: np.ndarray, 
        max_mser: float,
        min_area: int
    ) -> list[np.ndarray]:
        '''
        Performs a 3-step object subdivision based on a simplified MSERTree using node representatives, 
        filtering and ultimate erosion to determine the number of clusters in each flat zone of MSCT nodes.

                Parameters:
                        `candidates` (MultiScaleComponentTreeNode): MSCT node deemed as candidate
                        `msct` (MultiScaleComponentTree): MSCT to which candidates belong to
                        `image` (ndarray): image represented by the MSCT
                        `max_mser` (float): Maximum MSER value when traversing the tree
                        `min_area` (int): Minimum surface area of final connected components

                Returns:
                        `final_clusters` (list): list of clusters being boolean images
        '''
        # step 1 : simplify the MSCT into a MSERTree
        mser_tree = MSCTObjectDivider.compute_mser_tree(candidate)
        #mser_tree.save_dot(os.path.join('output', f"{candidate.get_id()}_mser"))
        #msct.save_dot_from_node(os.path.join('output', f"{candidate.get_id()}_msct"), candidate)

        # step 2 : shaping building and filtering (min-tree frmm the MSERTree)
        s = Shaping()
        ShapingTreeNode._idgen = 0
        MSCTShapingTreeNode._idgen = 0
        MSCTShapingTree._idgen = 0
        s.build_min_tree(mser_tree, candidate)
        st = s.create_shaping_tree()
        #st.save_dot(os.path.join('output', f"{candidate.get_id()}_shaping"))
        mser_nodes_filtered = st.filter_return_leq(max_mser)
        msct_nodes_filtered = MSCTObjectDivider.msct_nodes_from_filtered_msertree(mser_nodes_filtered)

        #for f in msct_nodes_filtered:
        #    fz = msct.get_local_flat_zone(f, 0)
        #    img = msct.build_node_content_image(image, fz)
        #    save_image(img, f"{candidate.get_id()}_{f.get_id()}_fz", 'output')

        #msct.save_dot_highlight_from_node(os.path.join('output', f"{candidate.get_id()}_filtered"), candidate, msct_nodes_filtered)
        #mser_tree.save_dot_highlight_from_node(os.path.join('output', f"{candidate.get_id()}_filtered_mser"), mser_tree.get_root(), mser_nodes_filtered)
        #objects = MSCTObjectDivider.subdivide_mser(mser_tree, max_mser=max_mser, min_area=min_area)

        #final_clusters = MSCTObjectDivider.perform_ultimate_erosion_and_partition(intermediate_objects, msct, image)
        #final_clusters = MSCTObjectDivider.perform_ultimate_erosion_and_partition(candidates, msct, image)
        final_clusters = MSCTObjectDivider.perform_ultimate_erosion_and_partition(msct_nodes_filtered, msct, image, min_area)
        return final_clusters
    
    #--------------------------------------------------------------------------

    @staticmethod
    def msct_nodes_from_filtered_msertree(mser_nodes_filtered: set[MSERNode]) -> set[MultiScaleComponentTreeNode]:
        msct_nodes_filtered = set()
        if len(mser_nodes_filtered) < 2:
            return set([list(mser_nodes_filtered)[0].get_link()])
        root = sorted(list(mser_nodes_filtered), key=lambda x: x.get_link().get_mser(), reverse=False)[0]
        to_process = []
        to_process.append(root)
        while len(to_process) > 0:
            node = to_process.pop(0)
            nb = 0
            if len(node.get_children()) == 0:
                 msct_nodes_filtered.add(node.get_link())
            else:
                for child in node.get_children():
                    if child in mser_nodes_filtered:
                        nb += 1
                if nb == len(node.get_children()):
                    for child in node.get_children():
                        to_process.append(child)
                else:
                    msct_nodes_filtered.add(node.get_link())
        return msct_nodes_filtered

    #--------------------------------------------------------------------------

    @staticmethod
    def compute_mser_tree(root: MultiScaleComponentTreeNode) -> MSERTree:
        '''
        Computes a MSERTree from a subtree of a MultiScaleComponentTree starting from root node.

                Parameters:
                        `root` (MultiScaleComponentTreeNode): root of the subtree

                Returns:
                        `mser_tree`: (MSERTree): computed MSERTree representing the given subtree
        '''
        # computing the local MSER minima for each branch
        all_minima = set()
        MSER.compute_mser_local_minima_per_branch_at_scale(
            scale=0, node=root, minima=all_minima, 
            local_minima=root, local_minima_mser=root.get_mser()
        )
        mser_tree = MSERTree()
        MSCTObjectDivider.generate_mser_tree_rec(root, None, mser_tree, all_minima, True)
        #mser_tree.save_dot(os.path.join('output', f"mser_{root.get_id()}"))
        return mser_tree
    
    #--------------------------------------------------------------------------
    
    @staticmethod
    def generate_mser_tree_rec(node: MultiScaleComponentTreeNode, parent: MSERNode, tree: MSERTree, minima: set[MultiScaleComponentTreeNode], root: bool) -> None:
        '''
        Recursive method to compute a simplified version of a MultiScaleComponentTree called a MSERTree.
        Each branch in the original MSCT will be represented by a single node being the highest ancestor of that branch.
        The value attributed to each branch representative is the smallest MSER value across said branch.

                Parameters:
                        `node` (MultiScaleComponentTreeNode): node of the MSCT to process
                        `parent` (MSERNode): parent of the current branch in the MSERTree
                        `tree` (MSERTree): MSERTree to build
                        `minima` (set): list of branch MSER minima
                        `root` (bool): whether the branch is the root branch of the MSCT/MSER tree

                Returns:
                        None, building `tree`
        '''
        target = node
        found = False
        rec_parent = parent
        while True:
            if not found:
                # target is the MSER minima of the current branch
                if target in minima:
                    found = True
                    rep = MSERNode(father=parent, value=target.get_mser(), link=node)
                    rep.set_id(target.get_id())
                    rec_parent = rep
                    tree.add_node(rep)
                    if parent is not None:
                        parent.add_child(rep)
                    if root:
                        tree.set_root(rep)
            # only one child here
            if len(target.get_children()) == 1:
                target = list(target.get_children())[0]
            else:
                break
        # branch has ended, can be either a leaf or the start of a subtree
        if len(target.get_children()) > 1:
            # recursive call on all children with target as father
            for child in target.get_children():
                MSCTObjectDivider.generate_mser_tree_rec(child, rec_parent, tree, minima, False)

    #--------------------------------------------------------------------------
    
    @staticmethod
    def subdivide_mser(tree: MSERTree, max_mser: float, min_area: int) -> set[MultiScaleComponentTreeNode]:
        '''
        MSERTree filtering to separate objects.
        The tree is traversed while nodes have a MSER value <= max_mser and a surface area >= min_size.

                Parameters:
                        `tree` (MSERTree): MSERTree to process
                        `max_mser` (float): maximum MSER value
                        `min_area` (int): minimum surface area to keep a node

                Returns:
                        `objects` (set): set of objects as MultiScaleComponentTreeNode
        '''
        to_process = []
        to_process.append(tree.root)
        objects = set()
        while len(to_process) > 0:
            node = to_process.pop(0)

            # case 1 : node has MSER <= max_mser
            if node.value <= max_mser:
                # case 1-1 : node is a leaf --> keep as object
                if len(node.children) == 0:
                    objects.add(node.link)

                # case 1-2 : node is not a leaf -> discarding, adding its children to to_process
                else:
                    nb = 0
                    for child in node.children:
                        if child.value <= max_mser:
                            nb += 1
                    # case 1-2-1 : at least one child is valid --> discarding, adding all children to to_process
                    if nb > 0:
                        for child in node.children:
                            to_process.append(child)
                    # case 1-2-2 : no child is valid --> keep node as an object
                    else:
                        objects.add(node.link)

            # case 2 : node has MSER > max_mser
            else:
                # case 2-1 : at least one child node has MSER <= max_mser --> discard node, add all children to to_process
                nb = 0
                for child in node.children:
                    if child.value <= max_mser:
                        nb += 1
                if nb > 0:
                    for child in node.children:
                        to_process.append(child)

                # case 2-2 : no child has MSER <= max_mser
                else:
                    # case 2-2-1 : area is > min_area --> keep as final object
                    if node.link.get_total_area() > min_area:
                        objects.add(node.link)  
                    # case 2-2-2 : area is <= min_area --> discard, too small

        return objects
    
    #--------------------------------------------------------------------------

    @staticmethod
    def fill_holes(image: np.ndarray) -> np.ndarray:
        '''
        Fills holes in a binary shape (scipy binary_fill_holes)

                Parameters:
                        `image` (ndarray): binary image (0, 255)

                Returns:
                        `filled` (ndarray): binary image (0, 255) with filled holes
        '''
        filled = binary_fill_holes((image / 255).astype(bool))
        filled = np.array(filled * 255, dtype=np.uint8)
        return filled
    
    #--------------------------------------------------------------------------

    @staticmethod
    def smooth_and_threshold(image: np.ndarray) -> np.ndarray:
        '''
        Smoothing (average 3x3) + Otsu thresholding
        '''
        # smooth by 3x3 average
        smoothed = np.array(convolve2d(image, np.ones((5,5)), 'same') / 25, dtype=np.uint8)

        # Otsu thresholding
        threshold = Otsu.compute_optimal_threshold(smoothed)
        thresholded = np.zeros(image.shape, image.dtype)
        thresholded[np.where(smoothed > threshold)] = 255
        return thresholded
    
    @staticmethod
    def extract_connected_components(image: np.ndarray) -> list[np.ndarray]:
        '''
        Returns as many images as there are connected components in `image`

                Parameters:
                        `image` (ndarray): input image

                Returns:
                        `ccs` (list): list of images, one per coonected component of `image`
        '''
        labels, nb_labels = label(image)
        ccs = []
        for index in range(0, nb_labels):
            l = index + 1
            coords = np.where(labels == l)
            img = np.zeros_like(image, dtype=np.uint8)
            img[coords] = 255
            ccs.append(img)
        return ccs

    #--------------------------------------------------------------------------
    
    @staticmethod
    def perform_ultimate_erosion_and_partition(
        objects: set[MultiScaleComponentTreeNode], 
        msct: MultiScaleComponentTree, 
        image: np.ndarray,
        min_area: int
    ) -> list[np.ndarray]:
        '''
        Performs an ultimate erosion of the flat zone of each given node to separate the data into individual objects.
        The number of clusters is determined by the number of connected components the ultimate erosion yields.
        The cluster' centers are the centroids of each connected component.

                Parameters:
                        `objects` (set): set of MultiScaleComponentTreeNode to process
                        `msct` (MultiScaleComponentTree): MSCT from which `objects` belogn to
                        `image` (ndarray): original image represented by the MSCT
                        `min_area` (int): minimum surface area of clusters

                Returns:
                        `all_objects` (list): list of clusters being boolean images
        '''
        all_objects = []
        for object in objects:
            fz = msct.get_local_flat_zone(object, 0)
            fz_img = msct.build_node_content_image(image, fz)
            fz_real = msct.reconstruct_subtree(object, image)
            smoothed = MSCTObjectDivider.smooth_and_threshold(fz_real)
            filled = MSCTObjectDivider.fill_holes(smoothed)
            ccs = MSCTObjectDivider.extract_connected_components(filled)
            
            for cc in ccs:
                ccs_4 = MSCTObjectDivider.ultimate_erosion_v1(cc, connectivity=[[0,1,0],[1,1,1],[0,1,0]])
                ccs_8 = MSCTObjectDivider.ultimate_erosion_v1(cc, connectivity=[[1,1,1],[1,1,1],[1,1,1]])
                c_4 = MSCTObjectDivider.find_object_centroids(ccs_4)
                c_8 = MSCTObjectDivider.find_object_centroids(ccs_8)

                # find common centroids
                distance = ndi.distance_transform_edt(fz_img)
                filtered_centroids = MSCTObjectDivider.filter_centroids(distance, c_4, c_8)
            
                clusters = MSCTObjectDivider.partitionize_data_watershed(distance, cc, filtered_centroids, min_area)
                for cluster in clusters:
                    all_objects.append(cluster)

        return all_objects

    #--------------------------------------------------------------------------

    @staticmethod
    def ultimate_erosion_v1(fz_img: np.ndarray, connectivity=[[0,1,0],[1,1,1],[0,1,0]]) -> list[np.ndarray]:
        '''
        Perform an ultimate erosion of the given flat zone.

                Parameters:
                        `fz_img` (ndarray): numpy array of the flat zone

                Returns:
                        `ccs` (list): list of individual connected component as a list of images
        '''
        bin_img = np.zeros((fz_img.shape[0], fz_img.shape[1]), dtype=bool)
        bin_img[np.where(fz_img != 0)] = True
        ccs = []
        ccs.append(bin_img)
        prev = bin_img
        while True:
            eroded = binary_erosion(prev, structure=[[0,1,0],[1,1,1],[0,1,0]], iterations=1)
            prev = eroded
            cc, nb_cc = label(eroded, structure=np.array(connectivity))
            if nb_cc == 0:
                break
            else:
                for index in range(0, nb_cc):
                    cur_cc = np.zeros_like(eroded)
                    cur_cc[np.where(cc == (index + 1))] = True
                    found = False
                    for i in range(0, len(ccs)):
                        if np.count_nonzero(np.logical_and(ccs[i], cur_cc)) > 0:
                            ccs[i] = cur_cc
                            found = True
                    if not found:
                        ccs.append(cur_cc)
        return ccs
    
    #--------------------------------------------------------------------------

    @staticmethod
    def ultimate_erosion_v2(fz_img: np.ndarray) -> list[np.ndarray]:
        '''
        Perform an ultimate erosion of the given flat zone.

                Parameters:
                        `fz_img` (ndarray): numpy array of the flat zone

                Returns:
                        `ccs` (list): list of individual connected component as a list of images
        '''
        bin_img = np.zeros((fz_img.shape[0], fz_img.shape[1]), dtype=bool)
        bin_img[np.where(fz_img != 0)] = True
        ccs = []
        ccs.append(bin_img)
        prev = bin_img
        while True:
            eroded = binary_erosion(prev, structure=[[0,1,0],[1,1,1],[0,1,0]], iterations=1)
            prev = eroded
            cc, nb_cc = label(eroded, structure=[[0,1,0],[1,1,1],[0,1,0]])
            if nb_cc == 0:
                break
            else:
                cur_ccs = []
                for index in range(0, nb_cc):
                    cur_cc = np.zeros_like(eroded)
                    cur_cc[np.where(cc == (index + 1))] = True
                    cur_ccs.append((cur_cc, np.count_nonzero(cur_cc)))
                # sort by decreasing size
                sorted_cur_ccs = sorted(cur_ccs, key=lambda x: x[1], reverse=True)
                new_ccs = []
                for sorted_cur_cc, _ in sorted_cur_ccs:
                    if len(new_ccs) == 0:
                        new_ccs.append(sorted_cur_cc)
                    else:
                        add = True
                        rows, cols = np.where(sorted_cur_cc != 0)
                        cc_row_centroid = round(np.sum(rows) / len(rows))
                        cc_col_centroid = round(np.sum(cols) / len(cols))
                        for added_cc in new_ccs:
                            rows, cols = np.where(added_cc != 0)
                            added_cc_row_centroid = round(np.sum(rows) / len(rows))
                            added_cc_col_centroid = round(np.sum(cols) / len(cols))
                            if pow((cc_row_centroid - added_cc_row_centroid), 2) + pow(cc_col_centroid - added_cc_col_centroid, 2) < 100:
                                add = False
                        if add:
                            new_ccs.append(sorted_cur_cc)

                for new_cc in new_ccs:
                    found = False
                    for i in range(0, len(ccs)):
                        if np.count_nonzero(np.logical_and(ccs[i], new_cc)) > 0:
                            ccs[i] = new_cc
                            found = True
                    if not found:
                        ccs.append(new_cc)
        return ccs

    #--------------------------------------------------------------------------
        
    @staticmethod
    def find_object_centroids(ccs: list[np.ndarray]) -> list[tuple[int, int]]:
        '''
        Computes the centroid of each connected component.

                Parameters:
                        `ccs` (list): list of images representing each one connected component (non-zero values)
        
                Returns:
                        `centroids` (list): set of centroids (row, column)
        '''
        centroids = []
        for cc in ccs:
            coords = np.where(cc != 0)
            rows = coords[0]
            cols = coords[1]
            if len(rows) > 0 or len(cols) >0:
                row_center = round(np.sum(rows) / len(rows))
                col_center = round(np.sum(cols) / len(cols))
                centroids.append((row_center, col_center))
        return centroids
    
    #--------------------------------------------------------------------------

    @staticmethod
    def filter_centroids(
        distances: np.ndarray, 
        centroids_4: list[tuple[int, int]], 
        centroids_8:list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        '''
        Fixes ultimate erosion's centroids based on the distance transform and returns the final centroids for watershed

                Parameters:
                        `distances` (ndarray): distance transform
                        `centroids_4` (list): list of centroids of clusters obtained with ultimate erosion with 4-neighbourhood (CC)
                        `centroids_8` (list): list of centroids of clusters obtained with ultimate erosion with 8-neighbourhood (CC)

                Returns:
                        `centroids` (list): list of centroids of final clusters to feed to the watershed algorithm
        '''

        s_cc_4 = [(i, distances[i]) for i in centroids_4]
        s_cc_8 = [(i, distances[i]) for i in centroids_8]

        # finding common centroids
        common_cc = []
        diff_cc = []
        for index_8 in range(0, len(s_cc_8)):
            row_8, col_8 = s_cc_8[index_8][0]
            d8 = s_cc_8[index_8][1]
            for index_4 in range(0, len(s_cc_4)):
                row_4, col_4 = s_cc_4[index_4][0]
                d4 = s_cc_4[index_4][1]
                if row_4 == row_8 and col_4 == col_8:
                    common_cc.append((row_4, col_4, d4))
        for index_8 in range(0, len(s_cc_8)):
            row_8, col_8 = s_cc_8[index_8][0]
            d8 = s_cc_8[index_8][1]
            if (row_8, col_8, d8) not in common_cc:
                diff_cc.append((row_8, col_8, d8))
        for index_4 in range(0, len(s_cc_4)):
            row_4, col_4 = s_cc_4[index_4][0]
            d4 = s_cc_4[index_4][1]
            if (row_4, col_4, d4) not in common_cc:
                diff_cc.append((row_4, col_4, d4))

        # filter common centroids if they violate the distance condition
        filtered_common_cc = []
        if len(common_cc) <= 1:
            filtered_common_cc = common_cc
        else:
            sorted_cc = sorted(common_cc, key=lambda x: x[2], reverse=True)
            filtered_common_cc.append((sorted_cc[0][0], sorted_cc[0][1],sorted_cc[0][2]))
            for row, col, d in common_cc:
                valid = 0
                for row_f, col_f, d_f in filtered_common_cc:
                    if pow(row - row_f, 2) + pow(col - col_f, 2) > pow(d_f, 2):
                        valid += 1
                # outside range of all final_cc
                if valid == len(filtered_common_cc):
                    filtered_common_cc.append((row, col, d))

        # keep or eliminate non common based on distance (from distance trasnform)
        final_cc = []
        for row, col, d in filtered_common_cc:
            final_cc.append((row, col, d))
        if len(diff_cc) > 0:
            for row, col, d in diff_cc:
                valid = 0
                for row_f, col_f, d_f in final_cc:
                    if pow(row - row_f, 2) + pow(col - col_f, 2) > pow(d_f, 2):
                        valid += 1
                # outside range of all final_cc
                if valid == len(final_cc):
                    final_cc.append((row, col, d))
        centroids = []
        for row, col, d in final_cc:
            centroids.append((row, col))
        return centroids

    #--------------------------------------------------------------------------
    
    @staticmethod
    def partitionize_data_kmeans(
        centers: list[tuple[int, int]], 
        pixels: list[tuple[int, int]]
    ) -> list[list[tuple[int, int]]]:
        '''
        Partitionize a flat zone using the Kmeans algorithm and the given centers as original centers.

                Parameters:
                        `centers` (list): list of estimated centers
                        `pixels` (list): list of pixels representing the flat zone

                Returns:
                        `clusters` (list): list of list of pixels (row, column), one for each cluster
        '''
        if len(centers) <= 1:
            return [pixels]
        else:        
            cluster_centers = np.array(centers)
            data = np.array(pixels)
            kmeans_init = KMeans(n_clusters=len(centers), init=cluster_centers, n_init=1)
            label = kmeans_init.fit_predict(data)
            u_labels = np.unique(label)
            clusters = []
            for i in u_labels:
                local_indices = np.where(label == i)
                local_pixels_list = []
                for local_index in local_indices:
                    local_pixels_list.append((data[local_index, 0], data[local_index, 1]))
                clusters.append(local_pixels_list)
            return clusters
        
    #--------------------------------------------------------------------------

    @staticmethod
    def partitionize_data_watershed(
        distance: np.ndarray, 
        fz: np.ndarray, 
        centroids: list[tuple[int, int]],
        min_area: int 
    ) -> list[np.ndarray]:
        '''
        Partitionizes the given flat-zone of node `node` using a watershed algorithm and the computed
        centroids in `centroids` as markers of said watershed.

                Parameters:
                        `distance` (ndarray): distance transform
                        `fz` (ndarray): flat-zone of the node (modified)
                        `centroids` (list): list of pixels (row, col) computed as centroids of segmentation classes
                        `min_area` (int): minimum surface area of connected components (classes)

                Returns:
                        `list_of_classes` (list): list of connected components as images
        '''

        # computing the distance transform and using centroids as markers for the watershed
        mask = np.zeros(distance.shape, dtype=bool)
        for row, col in centroids:
            mask[row, col] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=fz.astype(bool), watershed_line=False, compactness=0, connectivity=np.ones((3,3)))

        # creating the final segmented images
        big_classes = []
        small_classes = []

        for index in range(0, np.max(labels)):
            label = index + 1
            coords = np.where(labels == label)
            img = np.zeros_like(fz, dtype=bool)
            img[coords] = True
            if len(coords[0]) <= min_area:
                small_classes.append(img)
            else:
                big_classes.append(img)

        list_of_classes = MSCTObjectDivider.filter_watershed(big_classes, small_classes, fz, min_area)
        return list_of_classes

    #--------------------------------------------------------------------------

    @staticmethod
    def filter_watershed(big_ccs: list[np.ndarray], small_ccs: list[np.ndarray], image: np.ndarray, min_area: int) -> list[np.ndarray]:
        '''
        Filters connected components obtained by the watershed method.
        Attemps to merge connected components to their closest and biggest neighbour.
        If the sum of all connected components is under min_area, the merged CC is discarded

                Parameters:
                        `big_ccs` (list): list of big (> min_area) connected components (list of images)
                        `small_ccs` (list): list of small (<= min_area) connected components (list of images)
                        `image` (ndarray): flat zone image
                        `min_area` (int): minimum surface area in pixel for a cluster

                Returns:
                        `ccs` (list): filtered and merged list of connected components (list of images)
        '''
        list_of_classes = []

        # if all CC are small
        if len(big_ccs) == 0:
            area_all_smalls = 0
            for small_cc in small_ccs:
                area_small = len(np.where(small_cc == True)[0])
                area_all_smalls += area_small
            # merging all into a super CC or discarding all parts
            if area_all_smalls > min_area:
                merged_cc = np.zeros_like(image, dtype=bool)
                for small_cc in small_ccs:
                    merged_cc = np.logical_or(merged_cc, small_cc)
                list_of_classes.append(merged_cc)

        # using the big_ccs as base to merge the small_ccs
        else:
            
            list_of_classes = big_ccs

            for index_small in range(0, len(small_ccs)):
                cc_small = small_ccs[index_small]
                coords_small = np.where(cc_small == True)
                # find best neighbour (longest border)
                neighbours_and_size_per_class = []
                for index_big in range(0, len(list_of_classes)):
                    cc_big = list_of_classes[index_big]
                    coords_big = np.where(cc_big == True)
                    neighbours = 0
                    for pixel_i in coords_small:
                        local_neighbour = 0
                        if (pixel_i[0] - 1, pixel_i[1]) in coords_big:
                            local_neighbour += 1
                        if (pixel_i[0] + 1, pixel_i[1]) in coords_big:
                            local_neighbour += 1
                        if (pixel_i[0], pixel_i[1] - 1) in coords_big:
                            local_neighbour += 1
                        if (pixel_i[0], pixel_i[1] + 1) in coords_big:
                            local_neighbour += 1
                        if local_neighbour > 0:
                            neighbours += 1
                    neighbours_and_size_per_class.append((index_big, neighbours, len(coords_big[0])))
                # merge
                candidates = sorted(neighbours_and_size_per_class, key=lambda x:(x[1], x[2]), reverse=True)
                merge_index = candidates[0][0]
                list_of_classes[merge_index] = np.logical_or(list_of_classes[merge_index], cc_small)
        return list_of_classes       

    #--------------------------------------------------------------------------

    @staticmethod
    def identify_single_objects(msct: MultiScaleComponentTree, scale: int, image: np.ndarray, max_mser: float,min_area: int,debug=False) -> list[np.ndarray]:
        '''
        Attempts to divide objects among maximally augmented nodes.

                Parameters:
                        `msct` (MultiScaleComponentTree): global MSCT to process
                        `scale` (int): scale at which to perform computation (default=0, i.e. the highest)
                        `image` (ndarray): original image
                        `max_mser` (float): Minimum MSER value for objects
                        `min_area` (int): minimum surface area of clusters
                        `debug` (bool): whether to print debug informations

                Returns:
                        objects (list): list of single objects (boolean images)
        '''
        if debug:
            print(f"DEBUG {MSCTObjectDivider.identify_single_objects.__name__} : identifying single objects")

        width = image.shape[1]
        height = image.shape[0]

        # computing candidates for final objects (clusters)
        candidates = MSER.compute_mser_candidates_no_param(msct=msct, scale=scale, max_area=width*height, max_mser=max_mser, debug=debug)
        all_objects = []

        for candidate in candidates:
            candidate_objects = MSCTObjectDivider.compute_object_subdivision(candidate=candidate, msct=msct, image=image, max_mser=max_mser, min_area=min_area)
            for candidate_object in candidate_objects:
                all_objects.append(candidate_object)
        return all_objects
    
    #--------------------------------------------------------------------------
    
    @staticmethod
    def save_objects(
        image: np.ndarray, 
        objects: list[np.ndarray], 
        output_dir: str, 
        output_prefix='object',
        all=True,
        added_rows=0, 
        added_columns=0
    ) -> None:
        '''
        Saves one binary image for each identified object in `objects`.

                Parameters:
                        `image` (ndarray): original image used as template for dimensions
                        `objects` (list): list of clusters being boolean images
                        `output_dir` (str): output directory path
                        `output_prefix` (str): prefix to form an image filename as `output_prefix`_N.png
                        `all` (bool): whether to save all individual objects as images or just a combined colour image
                        `added_rows` (int): number of added rows to remove
                        `added_columns` (int): number of added columns to remove

                Returns:
                        None
        '''
        if not os.path.isdir(output_dir):
            Exception(f"{MSCTObjectDivider.save_objects.__name__} : {output_dir} is not a directory")
        
        h = image.shape[0] - added_rows
        w = image.shape[1] - added_columns
        all_objects_image = np.zeros((image.shape[0], image.shape[1], 3), image.dtype)
        colours = distinctipy.get_colors(len(objects))
        index = 0
        for obj in objects:
            fz_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            colour = colours[index]
            coords = np.where(obj == True)
            fz_image[coords] = 255          
            all_objects_image[coords] = np.array(colour) * 255
            if all:
                filename = f"{output_prefix}_{index}"
                fz_image = fz_image[0:h, 0:w]
                save_image(image=fz_image, name=filename, output_dir=output_dir)
            index += 1
        all_objects_image = all_objects_image[0:h, 0:w]
        save_image(image=all_objects_image, name='object_all', output_dir=output_dir)
