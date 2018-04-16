import logging
import os
import sys

import pydot_ng as pydot
from IPython.display import Image, display


def view_pydot(pydot_object):
    plt = Image(pydot_object.create_png())
    display(plt)


def create_graph(graph_info):
    dot = pydot.Dot()
    for node in graph_info['nodes']:
        dot.add_node(pydot.Node(node))
    for node1, node2 in graph_info['edges']:
        dot.add_edge(pydot.Edge(node1, node2))
    return dot


def view_graph(graph_info):
    graph = create_graph(graph_info)
    view_pydot(graph)


def plot_graph(graph_info, filepath):
    graph = create_graph(graph_info)
    graph.write(filepath, format='png')


def create_filepath(filepath):
    dirpath = os.path.dirname(filepath)
    os.makedirs(dirpath, exist_ok=True)


def initialize_logger():
    logger = logging.getLogger('steps')
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H-%M-%S')

    # console handler for validation info
    ch_va = logging.StreamHandler(sys.stdout)
    ch_va.setLevel(logging.INFO)

    ch_va.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(ch_va)

    return logger

def get_logger():
    return logging.getLogger('steps')