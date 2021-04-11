import os
import queue
import sys
import typing

import networkx as nx
from PIL import Image
from PIL import ImageDraw

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'

DUMMY_PREFIX = 'dummy-ca43f807-7738-4305-b56c-080ab5afa357-'

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
) -> typing.Tuple[typing.List[typing.List[str]], typing.Dict[str, int]]:
    layers: typing.List[typing.List[str]] = []
    vertex_layer: typing.Dict[str, int] = {}
    adj = graph.adj
    for vertex in reversed(vertices):
        min_layer = (
            (max(vertex_layer[v] for v in adj[vertex]) + 1)
            if adj[vertex]
            else 0
        )
        for i in range(min_layer, len(vertices)):
            if i >= len(layers):
                layers.append([])
            if len(layers[i]) < max_width:
                layers[i].append(vertex)
                vertex_layer[vertex] = i
                break

    return (
        [x for x in reversed(layers)],
        {
            key: (len(layers) - value - 1)
            for key, value in vertex_layer.items()
        },
    )


def build_fix_width(
        graph: nx.DiGraph, max_width: int,
) -> typing.Tuple[typing.List[typing.List[str]], typing.Dict[str, int]]:
    return fill_layers(top_sort(graph), graph, max_width)


def build_min_dummies(
        graph: nx.DiGraph,
) -> typing.Tuple[typing.List[typing.List[str]], typing.Dict[str, int]]:
    # todo
    return [], {}


def build_layers(
        graph: nx.DiGraph, max_width: typing.Optional[int],
) -> typing.Tuple[typing.List[typing.List[str]], typing.Dict[str, int]]:
    """
    Returns (layers with vertices, layer number by vertex map).
    """
    if max_width is None:
        return build_min_dummies(graph)
    return build_fix_width(graph, max_width)


def new_dummy(graph: nx.DiGraph, dummies_count: int) -> typing.Tuple[str, int]:
    dummy = DUMMY_PREFIX + str(dummies_count)
    dummies_count += 1
    graph.add_node(dummy)
    return dummy, dummies_count


def fill_dummies(
        graph: nx.DiGraph,
        layers: typing.List[typing.List[str]],
        vertex_layer: typing.Dict[str, int],
):
    adj = graph.adj
    dummies_count = 0
    for y, layer in enumerate(layers):
        for vertex in layer:
            if vertex.startswith(DUMMY_PREFIX):
                continue
            adj_vertices = list(adj[vertex].keys())
            for adj_vertex in adj_vertices:
                adj_y = vertex_layer[adj_vertex]
                if adj_y - y > 1:
                    graph.remove_edge(vertex, adj_vertex)
                    dummies = []
                    for i in range(adj_y - y - 1):
                        dummy, dummies_count = new_dummy(graph, dummies_count)
                        layers[y + i + 1].append(dummy)
                        vertex_layer[dummy] = y + i + 1
                        dummies.append(dummy)
                    vertices = [vertex] + dummies + [adj_vertex]
                    for v_from, v_to in zip(vertices[:-1], vertices[1:]):
                        graph.add_edge(v_from, v_to)


def visualize_graph(
        graph: nx.DiGraph,
        layers: typing.List[typing.List[str]],
        vertex_layer: typing.Dict[str, int],
        name: str,
):
    size = (max(len(layer) for layer in layers) - 1, len(layers) - 1)
    actual_size = ((size[0] + 2 * PAD) * X_UNIT, (size[1] + 2 * PAD) * Y_UNIT)
    image = Image.new('L', actual_size, 255)
    img_draw = ImageDraw.Draw(image)

    print(layers)

    adj = graph.adj
    for y, layer in enumerate(layers):
        for x, vertex in enumerate(layer):
            for adj_vertex in adj[vertex]:
                adj_y = vertex_layer[adj_vertex]
                adj_x = [
                    i for i, v in enumerate(layers[adj_y]) if v == adj_vertex
                ][0]
                img_draw.line(
                    (
                        (x + PAD) * X_UNIT,
                        (y + PAD) * Y_UNIT,
                        (adj_x + PAD) * X_UNIT,
                        (adj_y + PAD) * Y_UNIT,
                    ),
                    fill='black',
                    width=LINE_WIDTH,
                )
            if not vertex.startswith(DUMMY_PREFIX):
                img_draw.ellipse(
                    (
                        (x + PAD) * X_UNIT - DOT_SIZE / 2,
                        (y + PAD) * Y_UNIT - DOT_SIZE / 2,
                        (x + PAD) * X_UNIT + DOT_SIZE / 2,
                        (y + PAD) * Y_UNIT + DOT_SIZE / 2,
                    ),
                    fill='white',
                    outline='black',
                    width=LINE_WIDTH,
                )

    del img_draw
    image.save(os.path.join(OUTPUT_DIR, name + '.png'))


def main():
    max_width = input('W = ')
    max_width = int(max_width) if max_width else None

    for file_name in os.listdir(INPUT_DIR):
        if file_name.endswith('.xml') or file_name.endswith('.graphml'):
            path = os.path.join(INPUT_DIR, file_name)
            graph = nx.read_graphml(path)
            layers, vertex_layer = build_layers(graph, max_width)
            fill_dummies(graph, layers, vertex_layer)
            visualize_graph(graph, layers, vertex_layer, file_name)


if __name__ == '__main__':
    main()
