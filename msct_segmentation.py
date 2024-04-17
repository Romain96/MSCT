#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import numpy as np

from otsu import Otsu
from multi_scale_component_tree import MultiScaleComponentTree
from multi_scale_component_tree_node import MultiScaleComponentTreeNode
from mser import MSER
from mser_tree import MSERNode, MSERTree
from scipy.ndimage import binary_erosion, label
from scipy.signal import convolve2d
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes
from skimage.feature import peak_local_max

#------------------------------------------------------------------------------

def generate_mser_tree_rec(
    node: MultiScaleComponentTreeNode, 
    parent: MSERNode, 
    tree: MSERTree, 
    minima: set[MultiScaleComponentTreeNode], 
    root: bool
) -> None:
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
            generate_mser_tree_rec(child, rec_parent, tree, minima, False)

#------------------------------------------------------------------------------

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
    generate_mser_tree_rec(root, None, mser_tree, all_minima, True)
    return mser_tree

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------

def filter_mser_tree(tree: MSERTree, max_mser: float) -> set[MultiScaleComponentTreeNode]:

    to_process = [tree.get_root()]
    filtered = set()
    while len(to_process) > 0:
        node = to_process.pop(0)
        if node.get_value() <= max_mser:
            nb = 0
            for child in node.get_children():
                if child.get_value() <= max_mser:
                    nb += 1
            if nb == len(node.get_children()):
                for child in node.get_children():
                    to_process.append(child)
            else:
                filtered.add(node.get_link())
    if len(filtered) == 0:
        return set([tree.get_root().get_link()])
    else:
        return filtered  

#------------------------------------------------------------------------------

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
    
#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------

def perform_ultimate_erosion_and_partition(
    inter_cluster: MultiScaleComponentTreeNode, 
    msct: MultiScaleComponentTree, 
    image: np.ndarray,
    min_area: int
) -> list[np.ndarray]:
    '''
    Performs an ultimate erosion of the flat zone of each given node to separate the data into individual objects.
    The number of clusters is determined by the number of connected components the ultimate erosion yields.
    The cluster' centers are the centroids of each connected component.

            Parameters:
                    `inter_cluster` (MultiScaleComponentTreeNode): MultiScaleComponentTreeNode to process
                    `msct` (MultiScaleComponentTree): MSCT from which `objects` belogn to
                    `image` (ndarray): original image represented by the MSCT
                    `min_area` (int): minimum surface area of clusters

            Returns:
                    `all_objects` (list): list of clusters being boolean images
    '''
    all_objects = []
    fz = msct.get_local_flat_zone(inter_cluster, 0)
    fz_img = msct.build_node_content_image(image, fz)
    fz_real = msct.reconstruct_subtree(inter_cluster, image)
    smoothed = smooth_and_threshold(fz_real)
    filled = fill_holes(smoothed)
    ccs = extract_connected_components(filled)
            
    for cc in ccs:
        ccs_4 = ultimate_erosion(cc, connectivity=[[0,1,0],[1,1,1],[0,1,0]])
        ccs_8 = ultimate_erosion(cc, connectivity=[[1,1,1],[1,1,1],[1,1,1]])
        c_4 = find_object_centroids(ccs_4)
        c_8 = find_object_centroids(ccs_8)

        # find common centroids
        distance = ndi.distance_transform_edt(fz_img)
        filtered_centroids = filter_centroids(distance, c_4, c_8)

        clusters = partitionize_data_watershed(distance, cc, filtered_centroids, min_area)
        for cluster in clusters:
            all_objects.append(cluster)
    return all_objects

#------------------------------------------------------------------------------

def perform_ultimate_erosion_and_partition_direct(
    inter_cluster: MultiScaleComponentTreeNode, 
    msct: MultiScaleComponentTree, 
    image: np.ndarray
) -> list[np.ndarray]:
    '''
    Performs an ultimate erosion of the flat zone of each given node to separate the data into individual objects.
    The number of clusters is determined by the number of connected components the ultimate erosion yields.
    The cluster' centers are the centroids of each connected component.

            Parameters:
                    `inter_cluster` (MultiScaleComponentTreeNode): MultiScaleComponentTreeNode to process
                    `msct` (MultiScaleComponentTree): MSCT from which `objects` belogn to
                    `image` (ndarray): original image represented by the MSCT

            Returns:
                    `all_objects` (list): list of clusters being boolean images
    '''
    all_objects = []
    # reconstructing the flatzone from the cluster
    fz_real = msct.reconstruct_subtree(inter_cluster, image)
    # smoothing and tresholding (Otsu) to obtain finer detailed cluster(s)
    smoothed = smooth_and_threshold(fz_real)
    filled = fill_holes(smoothed)
    # extracting connected components : finer intermediary cluster(s)
    ccs = extract_connected_components(filled)
    
    for cc in ccs:
        # computing this distance transform (equivalent to an ultimate erosion)
        cluster_image = cc.astype(bool)
        distance = ndi.distance_transform_edt(cluster_image)
        max_distance = round(np.max(distance))
        # computing the peaks = local maxima
        # constraints = minimum distance between peaks of at least the radius of the biggest object
        #             = relative threshold value of 0.5 * max(image)
        coords = peak_local_max(distance, min_distance=max_distance, threshold_rel=0.5, footprint=np.ones((3, 3)), labels=cluster_image)
        mask = np.zeros(distance.shape, dtype=bool)
        # if no peak is found then the entire cluster is one single final cluster
        if len(coords) == 0:
            labels = np.zeros_like(cluster_image, dtype=np.uint8)
            labels[np.nonzero(cluster_image)] = 1
        # else a watershed algorithm is used on the cluster with the found peaks
        else:
            mask[tuple(coords.T)] = True
            markers, _ = ndi.label(mask)
            labels = watershed(-distance, markers, mask=cluster_image)
        # creating binary images, one per final cluster and adding to all_objects
        for label_index in range(1, np.max(labels) + 1):
            cluster = np.zeros_like(cluster_image, dtype=bool)
            cluster[np.where(labels == label_index)] = True
            all_objects.append(cluster)

    return all_objects

#------------------------------------------------------------------------------

def ultimate_erosion(fz_img: np.ndarray, connectivity=[[0,1,0],[1,1,1],[0,1,0]]) -> list[np.ndarray]:
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

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------

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

    list_of_classes = filter_watershed(big_classes, small_classes, fz, min_area)
    return list_of_classes

#------------------------------------------------------------------------------

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
            coords_small = np.nonzero(cc_small)
            # find best neighbour (longest border)
            neighbours_and_size_per_class = []
            for index_big in range(0, len(list_of_classes)):
                cc_big = list_of_classes[index_big]
                neighbours = 0
                for pixel_index in range(0, len(coords_small[0])):
                    pixel_i = (coords_small[0][pixel_index], coords_small[1][pixel_index])
                    local_neighbour = 0
                    if cc_big[pixel_i[0] - 1, pixel_i[1]] != 0:
                        local_neighbour += 1
                    if cc_big[pixel_i[0] + 1, pixel_i[1]] != 0:
                        local_neighbour += 1
                    if cc_big[pixel_i[0], pixel_i[1] - 1] != 0:
                        local_neighbour += 1
                    if cc_big[pixel_i[0], pixel_i[1] + 1] != 0:
                        local_neighbour += 1
                    if local_neighbour > 0:
                        neighbours += 1
                neighbours_and_size_per_class.append((index_big, neighbours, np.count_nonzero(cc_big)))
            # merge
            candidates = sorted(neighbours_and_size_per_class, key=lambda x:(x[1], x[2]), reverse=True)
            merge_index = candidates[0][0]
            list_of_classes[merge_index] = np.logical_or(list_of_classes[merge_index], cc_small)
    return list_of_classes          

#------------------------------------------------------------------------------

def segmentation_define_intermediate_clusters(
    candidate: MultiScaleComponentTreeNode, 
    max_mser: float
) -> list[MultiScaleComponentTreeNode]:
    # step 1 : simplify the MSCT into a MSERTree
    mser_tree = compute_mser_tree(candidate)

    # step 2 direct ffiltering of MSERTree
    msct_nodes_filtered = filter_mser_tree(mser_tree, max_mser)

    return msct_nodes_filtered

#------------------------------------------------------------------------------

def segmentation_define_final_clusters(
    node: MultiScaleComponentTreeNode,
    msct: MultiScaleComponentTree, 
    image: np.ndarray, 
    min_area: int
) -> list[np.ndarray]:

    # step 3 : ultimate erosion and watershed to obtain the final clusters
    #final_clusters = perform_ultimate_erosion_and_partition(node, msct, image, min_area)
    final_clusters = perform_ultimate_erosion_and_partition_direct(node, msct, image)
    return final_clusters

#------------------------------------------------------------------------------

def identify_single_objects(
    msct: MultiScaleComponentTree, 
    scale: int, 
    image: np.ndarray, 
    max_mser: float, 
    min_area: int
) -> list[np.ndarray]:
    '''
    Attempts to divide objects among maximally augmented nodes.

            Parameters:
                    `msct` (MultiScaleComponentTree): global MSCT to process
                    `scale` (int): scale at which to perform computation (default=0, i.e. the highest)
                    `image` (ndarray): original image
                    `max_mser` (float): Minimum MSER value for objects
                    `min_area` (int): minimum surface area of clusters

            Returns:
                    `objects` (list): list of single objects (boolean images)
    '''
    width = image.shape[1]
    height = image.shape[0]

    # computing candidates for final objects (clusters)
    candidates = MSER.compute_mser_candidates(msct=msct, scale=scale, max_area=width*height, max_mser=max_mser)
    all_objects = []

    intermediary_clusters = []
    for candidate in candidates:
        candidate_clusters = segmentation_define_intermediate_clusters(candidate, max_mser)
        for cluster in candidate_clusters:
            intermediary_clusters.append(cluster)
            #fz = msct.get_local_flat_zone(cluster, 0)
            #fz_img = msct.build_node_content_image(image, fz)
            #from msct_utils import save_image
            #save_image(fz_img, f"fz_{cluster.get_id()}", "output/flatzones")
    #msct.save_dot_highlight(f"output/msct/msct", intermediary_clusters, mode='dot')

    for inter_cluster in intermediary_clusters:
        candidate_clusters = segmentation_define_final_clusters(inter_cluster, msct, image, min_area)
        for cluster in candidate_clusters:
            all_objects.append(cluster)

    return all_objects
