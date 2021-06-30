import numpy as np


class Node:
    def __init__(self, left=None, right=None, counter=0, next_model_counter=0, split_attrib=0, split_value=0.0, depth=0):
        self.left = left
        self.right = right
        self.counter = counter
        self.next_model_counter = next_model_counter
        self.split_attrib = split_attrib
        self.split_value = split_value
        self.k = depth


def generate_max_min(dimensions):
    max_arr = np.zeros((dimensions))
    min_arr = np.zeros((dimensions))
    for q in range(dimensions):
        s_q = np.random.random_sample()
        max_value = max(s_q, 1 - s_q)
        max_arr[q] = s_q + 2 * max_value
        min_arr[q] = s_q - 2 * max_value
    return max_arr, min_arr


def build_single_hs_tree(max_arr, min_arr, current_depth, depth, num_dimensions):
    if current_depth == depth:
        return Node(depth=current_depth)
    node = Node()
    choosen_dim = np.random.randint(num_dimensions)
    split_threshold = (max_arr[choosen_dim] + min_arr[choosen_dim]) / 2.0
    temp = max_arr[choosen_dim]
    max_arr[choosen_dim] = split_threshold
    node.left = build_single_hs_tree(max_arr, min_arr, current_depth + 1, depth, num_dimensions)
    max_arr[choosen_dim] = temp
    min_arr[choosen_dim] = split_threshold
    node.right = build_single_hs_tree(max_arr, min_arr, current_depth + 1, depth, num_dimensions)
    node.split_attrib = choosen_dim
    node.split_value = split_threshold
    node.k = current_depth
    return node


def update_mass(x, node):
    if node:
        if node.k != 0:
            node.counter += 1
        if x[node.split_attrib] > node.split_value:
            node_new = node.right
        else:
            node_new = node.left
        update_mass(x, node_new)


def update_mass_next_model(x, node):
    if node:
        if node.k != 0:
            node.next_model_counter += 1
        if x[node.split_attrib] > node.split_value:
            node_new = node.right
        else:
            node_new = node.left
        update_mass_next_model(x, node_new)


def score_tree(x, node):
    if not node:
        return 0
    if x[node.split_attrib] > node.split_value:
        node_new = node.right
    else:
        node_new = node.left
    return node.counter * (2 ** node.k) + score_tree(x, node_new)


def score_tree_leaf_only(x, node):
    current_node = node
    while current_node.right and current_node.left:
        if x[current_node.split_attrib] > current_node.split_value:
            current_node = current_node.right
        else:
            current_node = current_node.left
    return current_node.counter


def update_model(node, weight_old=0.0, weight_new=1.0):
    if node:
        node.counter = node.counter * weight_old + node.next_model_counter * weight_new
        node.next_model_counter = 0
        update_model(node.left, weight_old, weight_new)
        update_model(node.right, weight_old, weight_new)


def build_trees(num_dimensions, num_trees, depth):
    hs_tree_list = []
    for i in range(num_trees):
        max_arr, min_arr = generate_max_min(num_dimensions)
        hs_tree_list.append(build_single_hs_tree(max_arr, min_arr, 0, depth, num_dimensions))
    return hs_tree_list

