import os
import queue
import sys
import typing

import networkx as nx
from PIL import Image
from PIL import ImageDraw

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'

X_UNIT = 40
Y_UNIT = 40
DOT_SIZE = 20
LINE_WIDTH = 2
PAD = 2  # in units


def insert_next_vertices(
        vertex: str,
        adj: nx.classes.coreviews.AdjacencyView,
        preds: nx.classes.coreviews.AdjacencyView,
        vertex_priority: typing.Dict[str, int],
        next_queue: queue.PriorityQueue,
        next_set: typing.Set[str],
):
    for next_vertex in adj[vertex]:
        next_preds = preds[next_vertex]
        if next_vertex not in next_set and all(
                (v in vertex_priority) for v in next_preds
        ):
            priority = sorted(vertex_priority[v] for v in next_preds)

            next_queue.put((priority, next_vertex))
            next_set.add(next_vertex)


def top_sort(graph: nx.DiGraph) -> typing.List[str]:
    result: typing.List[str] = []
    vertex_priority: typing.Dict[str, int] = {}
    next_queue: queue.PriorityQueue = queue.PriorityQueue()
    next_set: typing.Set[str] = set()

    preds = graph.pred
    adj = graph.adj
    for vertex in graph.nodes:
        if not preds[vertex]:
            vertex_priority[vertex] = len(result)
            result.append(vertex)

            insert_next_vertices(
                vertex, adj, preds, vertex_priority, next_queue, next_set,
            )

    assert next_set

    while next_set:
        _, vertex = next_queue.get()
        next_set.remove(vertex)

        vertex_priority[vertex] = len(result)
        result.append(vertex)
        insert_next_vertices(
            vertex, adj, preds, vertex_priority, next_queue, next_set,
        )
    return result


def fill_layers(
        vertices: typing.List[str], graph: nx.DiGraph, max_width: int,
) -> typing.List[typing.List[str]]:
    layers: typing.List[typing.List[str]] = []
    adj = graph.adj
    vertex_layer: typing.Dict[str, int] = {}
    for vertex in reversed(vertices):
        min_layer = (
            (min(vertex_layer[v] for v in adj[vertex]) + 1)
            if adj[vertex]
            else 0
        )
        for i in range(min_layer, len(vertices)):
            if i >= len(layers):
                layers.append([])
            if len(layers[i]) < max_width:
                layers[i].append(vertex)
                break

    return layers


def build_fix_width(
        graph: nx.DiGraph, max_width: int,
) -> typing.List[typing.List[str]]:
    return fill_layers(top_sort(graph), graph, max_width)


def build_min_dummies(graph: nx.DiGraph) -> typing.List[typing.List[str]]:
    # todo
    return []


def build_layers(
        graph: nx.DiGraph, max_width: typing.Optional[int],
) -> typing.List[typing.List[str]]:
    """
    Returns (layers with vertices, size of graph).
    Size is measured in units. Units must be converted to pixels
    by multiplying on X_UNIT or Y_UNIT.
    """
    if max_width is None:
        return build_min_dummies(graph)
    return build_fix_width(graph, max_width)


def visualize_graph(
        graph: nx.DiGraph, layers: typing.List[typing.List[str]], name: str,
):
    size = (max(len(layer) for layer in layers), len(layers))
    actual_size = ((size[0] + 2 * PAD) * X_UNIT, (size[1] + 2 * PAD) * Y_UNIT)
    image = Image.new('L', actual_size, 255)
    img_draw = ImageDraw.Draw(image)
    # todo
    del img_draw
    image.save(os.path.join(OUTPUT_DIR, name + '.png'))


def main():
    max_width = input('W = ')
    max_width = int(max_width) if max_width else None

    for file_name in os.listdir(INPUT_DIR):
        if file_name.endswith('.xml') or file_name.endswith('.graphml'):
            path = os.path.join(INPUT_DIR, file_name)
            graph = nx.read_graphml(path)
            layers = build_layers(graph, max_width)
            visualize_graph(graph, layers, file_name)


if __name__ == '__main__':
    main()
