import os
import sys
import typing

import networkx as nx
from PIL import Image
from PIL import ImageDraw

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'

MIN_DISTANCE = 2  # in units

X_UNIT = 40
Y_UNIT = 40
DOT_SIZE = 20
LINE_WIDTH = 2
PAD = 2  # in units


class Node:
    def __init__(self, x: int = 0, y: int = 0):
        self._x = x
        self._y = y
        self.x_shift = 0
        self.y_shift = 0
        self.children: typing.List[Node] = []

    @property
    def x(self):
        return self._x + self.x_shift

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def y(self):
        return self._y + self.y_shift

    @y.setter
    def y(self, y):
        self._y = y

    def draw(self, image: ImageDraw, x_shift: int = 0, y_shift: int = 0):
        self.x_shift += x_shift
        self.y_shift += y_shift
        for child in self.children:
            image.line(
                (
                    self.x * X_UNIT,
                    self.y * Y_UNIT,
                    child.x * X_UNIT + self.x_shift * X_UNIT,
                    child.y * Y_UNIT + self.y_shift * Y_UNIT,
                ),
                fill='black',
                width=LINE_WIDTH,
            )
            child.draw(image, self.x_shift, self.y_shift)
        image.ellipse(
            (
                self.x * X_UNIT - DOT_SIZE / 2,
                self.y * Y_UNIT - DOT_SIZE / 2,
                self.x * X_UNIT + DOT_SIZE / 2,
                self.y * Y_UNIT + DOT_SIZE / 2,
            ),
            fill='white',
            outline='black',
            width=LINE_WIDTH,
        )

    def shift(self, x_shift: int = 0, y_shift: int = 0):
        self.x_shift = x_shift
        self.y_shift = y_shift


def dfs_build(
        graph_adj: nx.classes.coreviews.AdjacencyView,
        graph_node: str,
        depth: int = 0,
) -> typing.Tuple[Node, typing.List[dict]]:
    """
    Returns (tree node, layers info).
    layers info: for each layer {
        'min': min x coord in layer,
        'max': max x coord in layer,
    }
    """
    node = Node(0, depth)
    layers_info: typing.List[dict] = [{}]

    children = graph_adj[graph_node].keys()
    prev_layers_info: typing.List[dict] = []
    for i, child in enumerate(children):
        child_node, child_layers_info = dfs_build(graph_adj, child, depth + 1)
        node.children.append(child_node)

        required_shifts = [
            max(
                MIN_DISTANCE
                - (child_layers_info[i]['min'] - prev_layers_info[i]['max']),
                0,
            )
            for i in range(min(len(prev_layers_info), len(child_layers_info)))
        ]
        shift = max(required_shifts) if required_shifts else 0
        child_node.shift(shift)

        layers_info += child_layers_info[len(layers_info[1:]) :]
        for j, info in enumerate(child_layers_info):
            info['max'] += shift
            layers_info[j + 1]['max'] = info['max']

        prev_layers_info[: len(child_layers_info)] = child_layers_info

    x = 0
    if children:
        x = (node.children[-1].x + node.children[0].x) // 2
    layers_info[0]['min'] = x
    layers_info[0]['max'] = x
    node.x = x

    return node, layers_info


def build_tree(
        graph: nx.DiGraph,
) -> typing.Tuple[Node, typing.Tuple[int, int]]:
    """
    Returns (tree root, size of tree).
    Tree size is measured in units. Units must be converted to pixels
    by multiplying on X_UNIT or Y_UNIT.
    """
    is_root_map = {node: True for node in graph.nodes}
    for _, node_to in graph.edges:
        is_root_map[node_to] = False
    roots = [node for node, is_root in is_root_map.items() if is_root]
    assert len(roots) == 1

    root = roots[0]
    tree, layers_info = dfs_build(graph.adj, root)

    min_x = sys.maxsize
    max_x = -sys.maxsize
    for info in layers_info:
        min_x = min(min_x, info['min'])
        max_x = max(max_x, info['max'])
    assert min_x == 0

    return tree, (max_x - min_x, len(layers_info))


def visualize_tree(tree: Node, name: str, size: typing.Tuple[int, int]):
    actual_size = ((size[0] + 2 * PAD) * X_UNIT, (size[1] + 2 * PAD) * Y_UNIT)
    image = Image.new('L', actual_size, 255)
    img_draw = ImageDraw.Draw(image)
    tree.shift(PAD, PAD)
    tree.draw(img_draw)
    del img_draw
    image.save(os.path.join(OUTPUT_DIR, name + '.png'))


def main():
    for file_name in os.listdir(INPUT_DIR):
        if file_name.endswith('.xml') or file_name.endswith('.graphml'):
            path = os.path.join(INPUT_DIR, file_name)
            tree, size = build_tree(nx.read_graphml(path))
            visualize_tree(tree, file_name, size)


if __name__ == '__main__':
    main()
