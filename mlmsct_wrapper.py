#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import os
import numpy as np
import matplotlib.pyplot as plt
import csv

from rcc8 import RCC8Relation, RCC8, rcc8_to_text
from msct_wrapper import MSCTWrapper
from multi_scale_component_tree import MultiScaleComponentTree
from multi_scale_component_tree_node import MultiScaleComponentTreeNode

#------------------------------------------------------------------------------

class MLMSCTWrapper():
    """
    This class represents a wrapper made to contain a multi-layer multi-scale component-tree (MLMSCT).

    A MLMSCT is a collection of multiple MSCT built on a multi-layer image.
    Each MSCT is built on one layer of the multi-layer image and connected together using RCC8 relations.
    """

    def __init__(self, image: np.ndarray):
        self.size = 0           # number of image layers / MSCT
        self.image = image      # multi-layer image
        self.mscts = list()     # MSCT for each layer
        self.relations = set()  # RCC8 relations between nodes of MSCTs

    #--------------------------------------------------------------------------

    def get_size(self) -> int:
        '''
        Getter for attribute `size`
        '''
        return self.size
    
    def set_size(self, size: int) -> None:
        '''
        Setter for attribute `size`
        '''
        self.size = size

    #----------------------------------------

    def get_image(self) -> np.ndarray:
        '''
        Getter for attribute `image`
        '''
        return self.image
    
    def set_image(self, image: np.ndarray) -> None:
        '''
        Setter for attribute `image`
        '''
        self.image = image

    #----------------------------------------

    def get_mscts(self) -> list:
        '''
        Getter for attribute `mscts`
        '''
        return self.mscts
    
    def set_mscts(self, mscts: list) -> None:
        '''
        Setter for attribute `mscts`
        '''
        self.mscts = mscts

    #----------------------------------------

    def get_relations(self) -> set:
        '''
        Getter for attribute `relations`
        '''
        return self.relations
    
    def set_relations(self, relations: set) -> None:
        '''
        Setter for attribute `relations`
        '''
        self.relations = relations

    #--------------------------------------------------------------------------

    def get_layer_data(self, index: int) -> MSCTWrapper:
        '''
        Returns wrapper data for the `index`th layer.

                Parameters:
                        `index` (int): index of the layer in the multi-layer image
                
                Returns:
                        `msct` (MSCTWrapper): MSCT wrapper of the `index`th layer
        '''
        if index < 0 or index >= self.get_size():
            raise Exception(f"ERROR in {self.get_layer_data.__name__} : no data for index {index}")
        msct = self.get_mscts()[index]
        return msct
    
    #--------------------------------------------------------------------------

    def add_layer_data(self, layer_image: np.ndarray, msct: MSCTWrapper) -> None:
        '''
        Adds data for the next layer

                Parameters:
                        `layer_image` (ndarray): gray-scale image for the processed layer
                        `msct` (MSCTWrapper) : MSCT for the processed layer image as wrapper (images, msct...)

                Returns:
                        None
        '''
        self.get_mscts().append(msct)
        self.set_size(self.get_size() + 1)

    #--------------------------------------------------------------------------

    def add_rcc8_relation(
            self, 
            index_a: int, 
            node_a: MultiScaleComponentTreeNode, 
            index_b: int, 
            node_b: MultiScaleComponentTreeNode, 
            rel_ab: RCC8Relation,
            rel_ba: RCC8Relation,
            percent_ab: float,
            percent_ba: float
            ) -> None:
        '''
        Adds a RCC8 relation `relation` from the node `node_a` in the layer `layer_a` to the node `node_b` in the layer `layer_b`.

                Parameters:
                        `index_a` (int): index of the layer to which the first node belong
                        `node_a` (MultiScaleComponentTreeNode): first node
                        `index_b` (int): index of the layer to which the second node belong
                        `node_b` (MultiScaleComponentTreeNode): second node
                        `rel_ab` (RCC8Relation): RCC8 relation from `node_a` to `node_b`
                        `rel_ba` (RCC8Relation): RCC8 relation from `node_b` to `node_a`
                        `percent_ab` (float): percent of common pixels of `node_a` inter `node_b` relative to `node_a`
                        `percent_ba` (float): percent of common pixels of `node_a` inter `node_b` relative to `node_b`
                
                Returns:
                        None
        '''
        self.relations.add((index_a, node_a, index_b, node_b, rel_ab, rel_ba, percent_ab, percent_ba))

    #--------------------------------------------------------------------------

    def connect_layers_rcc8(self) -> None:
        '''
        Connects nodes of layer's MSCT together to form a MLMSCT.
        Region Connection Calculus 8 (RCC8) is used to qualify the different types of relations.

                Parameters:
                        None

                Returns:
                        None
        '''
        # find the nodes to connect as a list of lists
        nodes_per_layer = list()
        for layer_index in range(0, self.get_size()):
            layer_nodes = list()
            msct_wrapper = self.get_mscts()[layer_index]
            layer_nodes = list(msct_wrapper.get_msct().get_augmented_nodes())
            nodes_per_layer.append(layer_nodes)

        # find all possible connections between them using RCC8
        msct_dapi = self.get_layer_data(0).get_msct()
        for node_dapi in nodes_per_layer[0]:
            fz_node_dapi = msct_dapi.get_flat_zone(node_dapi)
            for layer_index in range(1, self.get_size()):
                msct_other = self.get_layer_data(layer_index).get_msct()
                for node_other in nodes_per_layer[layer_index]:
                    fz_node_other = msct_other.get_flat_zone(node_other)
                    rel_dapi_other, rel_other_dapi, percent_dapi, percent_other = RCC8.compute_relation(fz_node_dapi, fz_node_other)
                    self.add_rcc8_relation(0, node_dapi, layer_index, node_other, rel_dapi_other, rel_other_dapi, percent_dapi, percent_other)
                    #self.add_rcc8_relation(0, node_dapi, layer_index, node_other, rel_dapi_other)
                    #self.add_rcc8_relation(layer_index, node_other, 0, node_dapi, rel_other_dapi)
                    #print(f"RCC8 betwee DAPI {hex(id(node_dapi))} and other ({layer_index}) {hex(id(node_other))} is {rel_dapi_other} {rel_other_dapi}")
                    if rel_dapi_other != RCC8Relation.DC or rel_other_dapi != RCC8Relation.DC:
                        msg = f"RCC8 between DAPI {node_dapi.get_id()} and other ({layer_index}) {node_other.get_id()} is :\n"
                        msg += f"{rel_dapi_other} ({percent_dapi * 100} %) {rel_other_dapi} ({percent_other * 100} %)"
                        print(msg)
                        #print(f"RCC8 betwee DAPI {node_dapi.get_id()} and other ({layer_index}) {node_other.get_id()} is {rel_dapi_other} {rel_other_dapi}")

    #--------------------------------------------------------------------------

    def visualize_relations(self, output_dir: str, exclude_disconnected=True) -> None:
        '''
        Exports images of both shapes and the computed RCC8 relation using Matplotlib's pyplot module
        '''
        for relation in self.get_relations():

            index_src, node_src, index_dst, node_dst, rel_ab, rel_ba, percent_ab, percent_ba = relation

            if exclude_disconnected and rel_ab == RCC8Relation.DC:
                continue
                
            msct_src = self.get_layer_data(index_src).get_msct()
            msct_dst = self.get_layer_data(index_dst).get_msct()

            shape_src = msct_src.get_flat_zone(node_src)
            shape_dst = msct_dst.get_flat_zone(node_dst)

            w = self.get_image().shape[0]
            h = self.get_image().shape[1]
            image_src = np.zeros((w, h), dtype=np.uint8)
            image_dst = np.zeros((w, h), dtype=np.uint8)

            for row, col in shape_src:
                image_src[row, col] = 255
            for row, col in shape_dst:
                image_dst[row, col] = 255

            name_src = f"layer_{index_src}_node_{node_src.get_id()}"
            name_dst = f"layer_{index_dst}_node_{node_dst.get_id()}"
            filename_ab = os.path.join(output_dir, f"{name_src}_{name_dst}.png")
            filename_ba = os.path.join(output_dir, f"{name_dst}_{name_src}.png")

            plt.cla()
            plt.clf()
            plt.close('all')
            fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6, 6))
            ax[0].imshow(image_src)
            ax[0].set_title(name_src)
            ax[1].imshow(image_dst)
            ax[1].set_title(name_dst)
            fig.suptitle(f"RCC8 - {rcc8_to_text(rel_ab)} ({percent_ab})")
            fig.savefig(filename_ab)

            plt.cla()
            plt.clf()
            plt.close('all')
            fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6, 6))
            ax[0].imshow(image_dst)
            ax[0].set_title(name_dst)
            ax[1].imshow(image_src)
            ax[1].set_title(name_src)
            fig.suptitle(f"RCC8 - {rcc8_to_text(rel_ba)} ({percent_ba})")
            fig.savefig(filename_ba)

    #--------------------------------------------------------------------------

    def export_relations_csv(self, output_dir: str, output_name: str) -> None:
        '''
        Exports all RCC8 relations to a CSV file.

                Parameters:
                        `output_dir` (str): output directory
                        `output_name` (str): output name without extension

                Returns:
                        None
        '''
        fields = ['layer_a', 'node_a', 'layer_b', 'node_b', 'rel_ab', 'rel_ba', 'percent_ab', 'percent_ba']
        filepath = os.path.join(output_dir, f"{output_name}.csv")

        with open(filepath, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            for layer_a, node_a, layer_b, node_b, rel_ab, rel_ba, percent_ab, percent_ba in self.get_relations():
                writer.writerow(
                    {
                        'layer_a': layer_a,
                        'node_a': node_a.get_id(),
                        'layer_b': layer_b,
                        'node_b': node_b.get_id(),
                        'rel_ab': rcc8_to_text(rel_ab),
                        'rel_ba': rcc8_to_text(rel_ba),
                        'percent_ab': percent_ab,
                        'percent_ba': percent_ba
                    }
                )
