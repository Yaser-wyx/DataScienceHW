import argparse
import networkx
import matplotlib.pyplot as plt
import random

"""
separate edge training set and edge test set.
In our experiment, we separate training set and test set from karate.edgelist file.
"""

# creating a parser
parser = argparse.ArgumentParser(description="run data processing to obtain graph training edges and  testing edges.")
parser.add_argument("--input_path", type=str, required=True, help="path to the original edgelist")
parser.add_argument("--output_train_path", type=str, required=True, help="save training graph to this path. ")
parser.add_argument("--output_test_path", type=str, required=True, help="save positive test edges to this path. ")
parser.add_argument("--testing_data_ratio", type=float, required=True, help="Ratio of edges to be used for testing.")


def node2vec_format(path, edges):
    """
    function that stores edges in a file in the format expected by node2vec algorithms.
    """
    with open(path, 'w') as file:
        for edge in edges:
            u, v = edge
            file.write(f"{u} {v}\n")
    print(f"Test edges stored in node2vec expected format, at: {path}")
    return None


def draw_graph(G, figname):
    """
    Given a graph, make a drawing and save figure with given name.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set(title=f'Graph for the {figname} dataset')
    networkx.draw(G, with_labels=False)
    fig.savefig(f'{figname}')
    print(f"Graph created!\n")


def createGraph(input_edge_list):
    """
    From list of edges create and return a graph.
    :param input_edge_list: list of edges
    :returns G: the graph
    """
    # first thing, how are the nodes separated in an edge
    with open(input_edge_list, 'r') as f:
        l = f.readline()
        delimiter = l[1]

    print(f"Graph creation started: ")
    G = networkx.read_edgelist(input_edge_list, delimiter=delimiter)
    print(f"----- Original graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    # only consider the largest connected component
    G = G.subgraph(max(networkx.connected_components(G), key=len)).copy()
    print(f"----- The largest component subgraph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.\n")

    return G


def split_train_test_graph(input_path, output_train_path, output_test_path, testing_data_ratio):
    """
    Function that samples test edges and removes them from original graph
    to obtain train_graph as well.
    :param input_path: path to original data
    :param output_train_path: path to save file with training edges
    :param output_test_path: path to file with test edges
    :param testing_data_ratio: ratio of edges to be used for testing
    """
    G = createGraph(input_path)  # create the original graph
    numEdges = G.number_of_edges()

    num_pos_edges = int(testing_data_ratio*numEdges)    # 向下取整
    test_positive_edges = random.sample(G.edges(), num_pos_edges)

    # removal of positive test edges:
    print(f"Edge removal started: ")
    G_train = G
    i = 0
    for edge in test_positive_edges:
        u, v = edge
        # if removing the edge does not create isolated nodes, then remove it
        if G_train.degree(u) > 1 and G_train.degree(v) > 1:
            G_train.remove_edge(u, v)
            i += 1

    # write out list of train edges that can be used for node embedding and classifier training
    networkx.write_edgelist(G_train, output_train_path, data=False)
    node2vec_format(output_test_path, test_positive_edges)  # store test positive edges
    print(f"----- Edge removal done. Training edges stored at {output_train_path}")
    return None


def main(input_path, output_train_path, output_test_path, testing_data_ratio=0.2):
    """
       main function that runs the whole process.
       :param input_path: path to original data
       :param output_train_path: path to save file with training edges
       :param output_test_path: path to file with test edges
       :param testing_data_ratio: ratio of edges to be used for testing
       """
    split_train_test_graph(input_path, output_train_path, output_test_path, testing_data_ratio)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.input_path, args.output_train_path, args.output_test_path, args.testing_data_ratio)