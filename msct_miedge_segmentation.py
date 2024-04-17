#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

#------------------------------------------------------------------------------

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import distinctipy
import matplotlib.pyplot as plt
import csv
from skimage.filters import threshold_otsu
import os
from ellipseRecognize import DTECMA

from multi_scale_component_tree import MultiScaleComponentTree
from multi_scale_component_tree_node import MultiScaleComponentTreeNode
from mser import MSER
from mser_tree import MSERNode, MSERTree
from scipy.ndimage import label, center_of_mass
from scipy.signal import convolve2d
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes
from skimage.feature import peak_local_max
from miedge_csv_utils import extract_dapi_from_annotations
from msct_utils import save_image

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
    filled = binary_fill_holes(image.astype(bool))
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
    threshold = int(round(threshold_otsu(smoothed)))
    thresholded = np.zeros(image.shape, image.dtype)
    thresholded[np.where(smoothed > threshold)] = 255
    return thresholded

#------------------------------------------------------------------------------

def morphological_closing(image: np.ndarray) -> np.ndarray:
    '''
    MM Closing on a binary flatzone.
    '''
    from scipy.ndimage import binary_closing
    from skimage.morphology import disk
    se = disk(radius=5)
    larger = np.zeros((image.shape[0]+10, image.shape[1]+10), dtype=int)
    larger[5:-5,5:-5] = image
    closed = binary_closing(larger, structure=se, border_value=1)
    return closed[5:-5,5:-5]

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
    ccs = extract_connected_components(smoothed)
    filled_ccs = []
    for cc in ccs:
        closed_cc = morphological_closing(cc)
        filled_cc = fill_holes(closed_cc)
        add = True
        for added_cc in filled_ccs:
            img_and = np.logical_and(cc.astype(bool), added_cc.astype(bool))
            if np.count_nonzero(img_and) > 0:
                added_cc = img_and.astype(int) * 255
                add = False
        if add:
            filled_ccs.append(filled_cc)
    
    for filled_cc in filled_ccs:
        # computing this distance transform (equivalent to an ultimate erosion)
        cluster_image = filled_cc.astype(bool)
        distance = ndi.distance_transform_edt(cluster_image)
        max_distance = round(np.max(distance))
        # computing the peaks = local maxima
        # constraints = minimum distance between peaks of at least the radius of the biggest object
        #             = relative threshold value of 0.5 * max(image)
        #coords = peak_local_max(distance, min_distance=max_distance, threshold_rel=0.5, footprint=np.ones((3, 3)), labels=cluster_image)
        coords = peak_local_max(distance, min_distance=7, exclude_border=False, labels=cluster_image)
        mask = np.zeros(distance.shape, dtype=bool)
        # if no peak is found then the entire cluster is one single final cluster
        if len(coords) <= 1:
            labels = np.zeros_like(cluster_image, dtype=np.uint8)
            labels[np.nonzero(cluster_image)] = 1
        # else a watershed algorithm is used on the cluster with the found peaks
        else:
            mask[tuple(coords.T)] = True
            markers, _ = ndi.label(mask)
            labels = watershed(-distance, markers, mask=cluster_image)

            '''fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
            ax = axes.ravel()

            ax[0].imshow(cluster_image, cmap=plt.cm.gray)
            ax[0].set_title('Overlapping objects')
            ax[1].imshow(-distance, cmap=plt.cm.gray)
            ax[1].plot(coords[:, 1],coords[:, 0],'rx')
            ax[1].set_title('Distances')
            ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
            ax[2].set_title('Separated objects')

            for a in ax:
                a.set_axis_off()

            fig.tight_layout()
            plt.show()'''

        for label_index in range(1, np.max(labels) + 1):
            cluster = np.zeros_like(cluster_image, dtype=bool)
            copy_coords = np.where(labels == label_index)
            if len(copy_coords[0]) > 0:
                cluster[copy_coords] = True
                all_objects.append(cluster)

    return all_objects

#------------------------------------------------------------------------------

def segment_with_dtecma(clusters: list[np.ndarray]) -> list[np.ndarray]:
    '''
    Test with Ellipse recognition DTECMA.
    '''
    all_objects = []

    nb = 1
    for cluster in clusters:
        print(f"\t\tcluster {nb}/{len(clusters)}")
        nb += 1
        ellipses = DTECMA(cluster, lam = 1.0, coverRate=0.75)
        centers = merge_close_ellipses(ellipses, 10)
        nuclei = dtecma_watershed_with_centers(image=cluster, centers=centers)
        for nucleus in nuclei:
            all_objects.append(nucleus)
    return all_objects

def merge_close_ellipses(ellipses: np.ndarray, max_distance: float) -> list[tuple[int, int]]:
    '''
    Merges all ellipses whose central points are closer to each other than a given threshold
    '''
    # get all ellipses in a list
    candidates = [(i[0], i[1]) for i in ellipses]
    # distance matrix
    dmat = np.zeros((len(candidates), len(candidates)), dtype=float)
    for i in range(0, len(candidates)):
        row_i, col_i = candidates[i]
        for j in range(0, len(candidates)):
            if i != j:
                row_j, col_j = candidates[j]
                dist = pow(row_i - row_j, 2) + pow(col_i - col_j, 2)
                dmat[i,j] = dist
                dmat[j,i] = dist
    # clustering
    indices = list(range(0, len(candidates)))
    to_merge = []
    limit = pow(max_distance, 2)
    while len(indices) > 0:
        index = indices.pop(0)
        cluster = [candidates[index]]
        remove_indices = []
        for candidate_index in indices:
            if dmat[index, candidate_index] <= limit:
                cluster.append(candidates[candidate_index])
                remove_indices.append(candidate_index)
        to_merge.append(cluster)
        for remove_index in remove_indices:
            indices.remove(remove_index)
    # each cluster will be represented by a new point being the center of mass of all points
    centers = []
    for merge_clusters in to_merge:
        if len(merge_clusters) == 1:
            centers.append((int(round(merge_clusters[0][0])), int(round(merge_clusters[0][1]))))
        else:
            row_com = 0
            col_com = 0
            for row, col in merge_clusters:
                row_com += row
                col_com += col
            center = (int(round(row_com / len(merge_clusters))), int(round(col_com / len(merge_clusters))))
            centers.append(center)
    return centers

def build_fake_distance_map_with_custom_centers(image: np.ndarray, centers: list[tuple[int,int]]) -> np.ndarray:
    distances = np.zeros_like(image, dtype=float)
    mask = np.nonzero(image)
    for i in range(0, len(mask[0])):
        row = mask[0][i]
        col = mask[1][i]
        dists = []
        for center in centers:
            d = pow(center[0] - row, 2) + pow(center[1] - col, 2)
            dists.append(d)
        dist = np.min(dists)
        distances[row, col] = dist
    return distances

#------------------------------------------------------------------------------

def dtecma_watershed_with_centers(image: np.ndarray, centers: list[tuple[int,int]]):
    cluster_image = image.astype(bool)
    #distance = ndi.distance_transform_edt(cluster_image)
    distance = build_fake_distance_map_with_custom_centers(image=image, centers=centers)
    mask = np.zeros(distance.shape, dtype=bool)
    all_objects = []

    # if no peak is found then the entire cluster is one single final cluster
    if len(centers) <= 1:
        labels = np.zeros_like(cluster_image, dtype=np.uint8)
        labels[np.nonzero(cluster_image)] = 1
        # else a watershed algorithm is used on the cluster with the found peaks
    else:
        for row, col in centers:
            mask[row, col] = True
        markers, _ = ndi.label(mask)
        labels = watershed(distance, markers, mask=cluster_image)

        '''fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(cluster_image, cmap=plt.cm.gray)
        ax[0].set_title('Overlapping objects')
        ax[1].imshow(-distance, cmap=plt.cm.gray)
        ax[1].plot([i[1] for i in centers], [i[0] for i in centers],'rx')
        ax[1].set_title('Distances')
        ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
        ax[2].set_title('Separated objects')

        for a in ax:
            a.set_axis_off()

        fig.tight_layout()
        plt.show()'''

    # creating nucleus images
    for label_index in range(1, np.max(labels) + 1):
        cluster = np.zeros_like(cluster_image, dtype=bool)
        copy_coords = np.where(labels == label_index)
        if len(copy_coords[0]) > 0:
            cluster[copy_coords] = True
            all_objects.append(cluster)
    return all_objects

#------------------------------------------------------------------------------

def compute_channel_vectors(nuclei_images: list[np.ndarray], channel_images: list[np.ndarray]) -> list[list[int]]:
    '''
    Computes the channel vectors of a set of pixels given a mask image of the set of pixels.

            Parameters:
                    `nuclei_images` (list): list of binary images for each nucleus
                    `channel_images` (list): list of channel images except the one used to build the MSCT

            Returns:
                    `data` (list): list of channel vectors composed from the channel images around each nucleus
    '''
    data = []
    for nucleus_image in nuclei_images:
        coords = np.where(nucleus_image > 0)
        hist = []
        for channel_image in channel_images:
            val = np.sum(channel_image[coords])
            hist.append(val / len(coords[0]))
        data.append(hist)
    return data

#------------------------------------------------------------------------------

def save_segmentation_as_label_image(data: list[np.ndarray], ref_image: np.ndarray, output_dir: str, output_name: str) -> None:
    '''
    Saves a label image with each label being a segmented nucleus as a numpy array of integer labels.

            Parameters:
                    `data` (list): list of binary nuclei images
                    `ref_image` (ndarray): reference image
                    `output_dir` (str): output directory path
                    `output_name` (str): filename to be created without extension

            Returns: None
    '''
    # make a label image
    label_img = np.zeros_like(ref_image, dtype=int)
    label_index = 1
    for nucleus_img in data:
        label_img[np.where(nucleus_img > 0)] = label_index
        label_index += 1
    # saving
    outfile = os.path.join(output_dir, f"{output_name}.npy")
    print(f"saving label image as {outfile}")
    np.save(outfile, label_img)

#------------------------------------------------------------------------------

def save_channel_vectors_as_array(data: list[list[int]], output_dir: str, output_name: str) -> None:
    '''
    Saves all channel vectors as a numpy array (matrix of (nb_nuclei, size_of_vector)).

            Parameters:
                    `data` (list): list of channel vectors for all nuclei
                    `output_dir` (str): output directory path
                    `output_name` (str): filename to be created without extension
    '''
    all_vectors = []
    for channel_vector in data:
        all_vectors.append(channel_vector)
    all_vectors = np.array(all_vectors)
    outfile = os.path.join(output_dir, f"{output_name}.npy")
    print(f"saving channel vectors as {outfile}")
    np.save(outfile, all_vectors)

#------------------------------------------------------------------------------

def segmentation_compute_and_filter_msct(candidate: MultiScaleComponentTreeNode, max_mser: float) -> list[MultiScaleComponentTreeNode]:
    # step 1 : simplify the MSCT into a MSERTree
    mser_tree = compute_mser_tree(candidate)

    # step 2 direct ffiltering of MSERTree
    msct_nodes_filtered = filter_mser_tree(mser_tree, max_mser)

    return msct_nodes_filtered

#------------------------------------------------------------------------------

def segmentation_clean_flatzones(nodes: list[np.ndarray], msct: MultiScaleComponentTree, image: np.ndarray,min_area: int) -> list[np.ndarray]:
    all_cleaned = []
    for node in nodes:
        fz_real = msct.reconstruct_subtree(node, image)
        smoothed_cc = smooth_and_threshold(fz_real)
        closed_cc = morphological_closing(smoothed_cc)
        filled_cc = fill_holes(closed_cc)
        ccs = extract_connected_components(filled_cc)
        '''
        for cc in ccs:
            i = 0
            stop = False
            merged = False
            coords = np.where(cc > 0)
            cc_coords = list(zip(coords[0], coords[1]))
            while i < len(all_cleaned) and not stop:
                # merge because overlap
                coords = np.where(all_cleaned[i] > 0)
                cleaned_coords = list(zip(coords[0], coords[1]))
                inter = set(cc_coords).intersection(set(cleaned_coords))
                if len(inter) > 0:
                    merge = all_cleaned[i]
                    merge[np.where(cc > 0)] = 255
                    all_cleaned[i] = merge
                    stop = True
                    merged = True
                i += 1
            # no overlap -> merged
            if not merged:
                all_cleaned.append(cc)'''
        for cc in ccs:
            if np.count_nonzero(cc) >= min_area:
                all_cleaned.append(cc)
    return all_cleaned

#------------------------------------------------------------------------------

def segmentation_define_final_clusters(nodes: list[MultiScaleComponentTreeNode]) -> list[np.ndarray]:

    nuclei = segment_with_dtecma(nodes)
    return nuclei

#------------------------------------------------------------------------------

def identify_single_objects(
    msct: MultiScaleComponentTree, 
    scale: int, 
    image: np.ndarray, 
    max_mser: float, 
    max_area_factor: float,
    min_area: int,
    annotations: dict,
    channel_images: list[np.ndarray]
) -> tuple[list[np.ndarray], list[list[int]]]:
    '''
    Attempts to divide objects among maximally augmented nodes.

            Parameters:
                    `msct` (MultiScaleComponentTree): global MSCT to process
                    `scale` (int): scale at which to perform computation (default=0, i.e. the highest)
                    `image` (ndarray): original image
                    `max_mser` (float): Minimum MSER value for objects
                    `max_area_factor` (float): maximum surface area when choosing nodes (% of image area)
                    `min_area` (int): minimum size of flatzone to be considered a nucleus
                    `annotations` (dict): dot annotations for DAPI
                    `channel_images` (list): list of channel images
            Returns:
                    `all_objects` (list): list of segmented nuclei
                    `channel_vectors` (list): list of channel vectors for each nucleus
    '''
    width = image.shape[1]
    height = image.shape[0]
    seg_max_area = width * height * max_area_factor

    # computing candidates for final objects (clusters)
    candidates = MSER.compute_mser_candidates(
        msct=msct, 
        scale=scale, 
        max_area=seg_max_area, 
        max_mser=max_mser
    )

    intermediary_clusters = []
    for candidate in candidates:
        candidate_clusters = segmentation_compute_and_filter_msct(candidate, max_mser)
        for cluster in candidate_clusters:
            intermediary_clusters.append(cluster)

    cleaned_ccs = segmentation_clean_flatzones(intermediary_clusters, msct, image, min_area)
    '''i = 0
    for cc in cleaned_ccs:
        save_image(cc, f"fz_{i}", 'output/flatzones')
        i+=1'''
    all_objects = segmentation_define_final_clusters(cleaned_ccs)

    # channel vectors and label image
    channel_vectors = compute_channel_vectors(nuclei_images=all_objects, channel_images=channel_images)
    
    return (all_objects, channel_vectors)
