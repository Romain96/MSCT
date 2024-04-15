#!/usr/bin/env python
__author__ = "Romain PERRIN"
__maintainer__ = "Romain PERRIN"
__email__ = "romain.perrin@unistra.fr"
__status__ = "Development"
__copyright__ = "Copyright 2023, Romain PERRIN, SDC Team, ICube UMR 7357, University of Strasbourg"

import pydot

#------------------------------------------------------------------------------

class MSERNode:

    def __init__(self, father, value, link) -> None:
        self.id = 0
        self.father = father
        self.children = set()
        self.value = value
        self.link = link

    #----------------------------------------

    def get_id(self) -> int:
        return self.id
    
    def set_id(self, id: int) -> None:
        self.id = id

    #----------------------------------------

    def get_father(self):
        return self.father
    
    def set_father(self, father) -> None:
        self.father = father

    #----------------------------------------

    def get_children(self):
        return self.children
    
    def set_children(self, children) -> None:
        self.children = children

    #----------------------------------------

    def get_value(self):
        return self.value
    
    def set_value(self, value) -> None:
        self.value = value

    #----------------------------------------

    def get_link(self):
        return self.link
    
    def set_link(self, link) -> None:
        self.link = link

    #----------------------------------------

    def add_child(self, child) -> None:
        self.children.add(child)

    def remove_child(self, child) -> None:
        self.children.remove(child)

    #----------------------------------------

    def debug(self) -> None:
        print(f"{MSERNode.__class__.__name__} : ID {hex(id(self))}")
        print(f"{MSERNode.__class__.__name__} : Father {hex(id(self.father))}")
        print(f"{MSERNode.__class__.__name__} : Value {self.value}")
        print(f"{MSERNode.__class__.__name__} : Link {hex(id(self.link))}")

#------------------------------------------------------------------------------

class MSERTree:

    def __init__(self) -> None:
        self.root = None
        self.nodes = set()

    #----------------------------------------

    def get_root(self) -> MSERNode:
        return self.root

    def set_root(self, root: MSERNode) -> None:
        self.root = root

    #----------------------------------------

    def get_nodes(self) -> set[MSERNode]:
        return self.nodes

    def set_nodes(self, nodes: set[MSERNode]) -> None:
        self.nodes = nodes

    #----------------------------------------

    def add_node(self, node: MSERNode) -> None:
        self.nodes.add(node)

    def remove_node(self, node: MSERNode) -> None:
        self.nodes.remove(node)

    #----------------------------------------

    def debug(self) -> None:
        print(f"{MSERNode.__class__.__name__} : ID {hex(id(self))}")
        print(f"{MSERNode.__class__.__name__} : Root {hex(id(self.root))}")
        print(f"{MSERNode.__class__.__name__} : Nodes {len(self.nodes)}")

    #----------------------------------------

    def save_dot(self, filename: str) -> None:
        self.save_dot_from_node(filename, self.root)

    #----------------------------------------

    def save_dot_from_node(self, filename: str, node: MSERNode) -> None:

        graph = pydot.Dot('MSERTree', graph_type='graph', bgcolor='white')

        # add all nodes from node to its subtree leaves
        process = set()
        process.add(node)

        while len(process) > 0:
            cur_node = process.pop()
            
            nid = hex(id(cur_node))
            nval = cur_node.value
            label_text = f"id {nid}\n{nval}\n({cur_node.link.get_id()})"
            gnode = pydot.Node(name=f"node_{nid}", label=label_text)
            graph.add_node(gnode)
            
            for child in cur_node.children:
                process.add(child)

        # connect nodes
        process = set()
        process.add(node)

        while len(process) > 0:
            cur_node = process.pop()
            for child in cur_node.children:
                process.add(child)
                nid = hex(id(cur_node))
                cid = hex(id(child))
                edge = pydot.Edge(f"node_{nid}", f"node_{cid}")
                graph.add_edge(edge)

        graph.write_png(f"{filename}.png")

    #----------------------------------------

    def save_dot_highlight_from_node(self, filename: str, node: MSERNode, nodes: set[MSERNode]) -> None:

        graph = pydot.Dot('component_tree', graph_type='graph', bgcolor='white')

        # add all nodes from node to its subtree leaves
        process = set()
        process.add(node)

        while len(process) > 0:
            cur_node = process.pop()

            nid = hex(id(cur_node))
            nval = cur_node.value
            label_text = f"id {nid}\n{nval}\n({cur_node.link.get_id()})"

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

        graph.write_png(f"{filename}.png")

#------------------------------------------------------------------------------
