import dhg
import random


# Classic Hypergraphs Generator:
# Temperal classic hypergraph: 6
def hyper_flower_builder(num_edges, num_center, num_unoverlap):
    """
    Building up a hyper-flower hypergraph, every hyperedge share the same nodes.
    num_edges: number of hyperedges
    num_center: number of nodes in the center
    num_unoverlap: number of nodes be special in single hyperedge
    """
    num_nodes = num_edges * num_unoverlap + num_center
    hyperedges = []
    common_set = [num_nodes - i - 1 for i in range(num_center)]
    for i in range(num_edges):
        node_list = [j + i * num_unoverlap for j in range(num_unoverlap)]
        node_list.extend(common_set)
        hyperedges.append(tuple(node_list))
    hyperGraph = dhg.Hypergraph(num_nodes, hyperedges)

    return num_nodes, hyperedges, shuffle_HG(hyperGraph)


def hyper_wheel_builder(num_edges, num_center, num_overlap, num_unoverlap):
    """
    Building up a hyper-wheel hypergraph, center nodes are in all hyperedges,
    two hyperedges share specific number of nodes.
    num_edges: number of hyperedges
    num_center: number of center nodes
    num_overlap: number of nodes two edges share; if num_overlap = 0, then it is the same as hyperflower
    num_unoverlap: number of nodes be special in single hyperedge
    """
    num_nodes = num_edges * (num_unoverlap + num_overlap) + num_center
    hyperedges = []
    common_set = [num_nodes - i - 1 for i in range(num_center)]
    start_list = [i * (num_unoverlap + num_overlap) for i in range(num_edges)]
    for i in range(num_edges):
        node_start = start_list[i]
        node_list = [
            (j + node_start) % (num_edges * (num_unoverlap + num_overlap))
            for j in range(num_unoverlap + num_overlap + num_overlap)
        ]

        node_list.extend(common_set)
        hyperedges.append(tuple(node_list))
    hyperGraph = dhg.Hypergraph(num_nodes, hyperedges)

    return num_nodes, hyperedges, shuffle_HG(hyperGraph)


def hyper_cycle_builder(num_edges, num_overlap, num_unoverlap):
    """
    Building up a hyper-wheel hypergraph, no center nodes,
    two hyperedges share specific number of nodes.
    num_edges: number of hyperedges
    num_overlap: number of nodes two edges share; if num_overlap = 0, then it is the same as hyperflower
    num_unoverlap: number of nodes be special in single hyperedge
    """
    num_nodes = num_edges * (num_unoverlap + num_overlap)
    hyperedges = []
    start_list = [i * (num_unoverlap + num_overlap) for i in range(num_edges)]
    for i in range(num_edges):
        node_start = start_list[i]
        node_list = [
            (j + node_start) % (num_edges * (num_unoverlap + num_overlap))
            for j in range(num_unoverlap + num_overlap + num_overlap)
        ]

        hyperedges.append(tuple(node_list))
    hyperGraph = dhg.Hypergraph(num_nodes, hyperedges)

    return num_nodes, hyperedges, shuffle_HG(hyperGraph)


def hyper_chain_builder(num_edges, num_overlap, num_unoverlap):
    """
    Building up a hyper-chain hypergraph, two hyperedges share num_overlap number of nodes, no loop.
    num_edges: number of hyperedges
    num_overlap: number of nodes two edges share; if num_overlap = 0, then it is the same as hyperflower
    num_unoverlap: number of nodes be special in single hyperedge
    """
    num_nodes = num_edges * (num_unoverlap + num_overlap) + num_overlap
    hyperedges = []
    start_list = [i * (num_unoverlap + num_overlap) for i in range(num_edges)]
    for i in range(num_edges):
        node_start = start_list[i]
        node_list = [
            (j + node_start) for j in range(num_unoverlap + num_overlap + num_overlap)
        ]

        hyperedges.append(tuple(node_list))
    hyperGraph = dhg.Hypergraph(num_nodes, hyperedges)

    return num_nodes, hyperedges, shuffle_HG(hyperGraph)


def hyper_fern_builder(num_overlap, num_unoverlap):
    """
    Building up a hyper-fern hypergraph, two hyperedges share one overlap node,
    all overlap nodes are connected in one hyperedge, no loop.
    --> #edges = num_overlap * 2 + 1
    num_overlap: total number of nodes two edges share;
    num_unoverlap: number of nodes be special in single hyperedge
    """
    num_nodes = num_overlap * num_unoverlap * 2 + num_overlap
    hyperedges = []
    overlap_list = [i + num_unoverlap * num_overlap * 2 for i in range(num_overlap)]
    for i in range(num_overlap):
        node_list_l = [j + i * num_unoverlap for j in range(num_unoverlap)]
        node_list_r = [
            j + i * num_unoverlap + num_overlap * num_unoverlap
            for j in range(num_unoverlap)
        ]
        node_list_l.append(overlap_list[i])
        node_list_r.append(overlap_list[i])
        hyperedges.append(tuple(node_list_l))
        hyperedges.append(tuple(node_list_r))
    hyperedges.append(tuple(overlap_list))
    hyperGraph = dhg.Hypergraph(num_nodes, hyperedges)

    return num_nodes, hyperedges, shuffle_HG(hyperGraph)


def hyper_lattice_builder(l):
    """
    Building up a Lattice/Grid hypergraph,
    a square web-structure with l^2 nodes, and 2l hyperedges.
    --> #edges = num_overlap * 2 + 1
    """
    num_nodes = l * l
    hyperedges = []

    for i in range(l):
        # Add horizontal list
        hyperedges.append(tuple([j + i * l for j in range(l)]))
        # Add vertical list
        hyperedges.append(tuple([l * j + i for j in range(l)]))
    hyperGraph = dhg.Hypergraph(num_nodes, hyperedges)

    return num_nodes, hyperedges, shuffle_HG(hyperGraph)


def shuffle_HG(graph):
    # Shuffle the hypergraph's edges and vectors
    shuffled = list(graph.v)
    random.shuffle(shuffled)
    v_map = {i: shuffled[i] for i in range(len(graph.v))}
    edge_list = list(graph.e[0])
    result = [tuple([v_map[j] for j in i]) for i in edge_list]
    return dhg.Hypergraph(len(graph.v), result)


# ------------------------------------------------------------------
# Generate Random dataset
def link_hypergraphs(graph1, graph2):
    """
    Link two hypergraphs into one, by simplying put two nodes together.
    """
    hyperedge_result = list(graph1.e[0])
    edge_2 = graph2.e[0]
    total_nodes = len(graph1.v) + len(graph2.v) - 1
    # Randomly pick a node from graph1:
    node1 = random.randint(0, len(graph1.v) - 1)
    # Ramdomly pick a node from graph 2:
    node2 = random.randint(0, len(graph2.v) - 1)
    # Reform edges into one
    for edge in edge_2:
        new_edge = []
        for node in edge:
            if node < node2:
                new_edge.append(node + len(graph1.v))
            elif node == node2:
                new_edge.append(node1)
            else:
                new_edge.append(node + len(graph1.v) - 1)
        hyperedge_result.append(tuple(new_edge))

    hyperGraph = dhg.Hypergraph(total_nodes, hyperedge_result)
    return hyperGraph


def random_n_graphs(generage_way="multi_class"):
    """
    Randomly build a data and its target vector for multi-class classification.
    Randomly picking a hypergraph from the list of classical hypergraphs and link them together.
    """
    # Randomly build a target vector to direct hypergraph's constructing
    target = []  # target vector
    num_graph_type = 6
    if generage_way == "multi_class":  # Multi-class classification data
        while sum(target) == 0:
            target = [random.randint(0, 1) for i in range(num_graph_type)]
    elif generage_way == "single_class":  # Single-class classification data
        target = [0] * num_graph_type
        target[random.randint(0, 5)] = 1

    # Generate hypergraphs and save into a list:
    saver = []
    if target[0] != 0:
        num_nodes, hyperedges, hypergraph = hyper_flower_builder(
            random.randint(3, 8), random.randint(1, 6), random.randint(1, 6)
        )
        saver.append(hypergraph)
    if target[1] != 0:
        num_nodes, hyperedges, hypergraph = hyper_wheel_builder(
            random.randint(3, 8),
            random.randint(1, 6),
            random.randint(1, 4),
            random.randint(1, 4),
        )
        saver.append(hypergraph)
    if target[2] != 0:
        num_nodes, hyperedges, hypergraph = hyper_cycle_builder(
            random.randint(3, 8), random.randint(1, 3), random.randint(1, 3)
        )
        saver.append(hypergraph)
    if target[3] != 0:
        num_nodes, hyperedges, hypergraph = hyper_chain_builder(
            random.randint(3, 8), random.randint(1, 3), random.randint(1, 3)
        )
        saver.append(hypergraph)
    if target[4] != 0:
        num_nodes, hyperedges, hypergraph = hyper_fern_builder(
            random.randint(1, 7), random.randint(1, 4)
        )
        saver.append(hypergraph)
    if target[5] != 0:
        num_nodes, hyperedges, hypergraph = hyper_lattice_builder(random.randint(2, 5))
        saver.append(hypergraph)

    # Link all hypergraphs in the list together
    if len(saver) == 1:
        result = saver[0]
        return target, result
    elif len(saver) > 1:
        result = link_hypergraphs(saver[0], saver[1])
        for i in range(len(saver) - 2):
            result = link_hypergraphs(hypergraph, saver[i + 2])
        return target, result


def data_Generator(n, generage_way="multi_class"):
    """
    Generates n graphs,
    """
    # Set seed
    seed = 5
    random.seed(seed)
    num_graph_type = 6
    data = []
    for i in range(n):
        target, Hgraph = random_n_graphs(generage_way)
        data.append((target, Hgraph))

    # Saving data into file in this form:
    # num_graphs \n
    # num_nodes, num_edges, graph_label \n
    # node_label_list \n
    # Edge’s_nodes_list \n
    with open(
        "{graph_number}Random_{num_graph_type}_hg_seed_{seed}_{generage_way}.txt".format(
            num_graph_type=num_graph_type,
            seed=seed,
            generage_way=generage_way,
            graph_number=n,
        ),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(n) + "\n")
        for item in data:
            (target, graph) = item
            f.write(
                str(len(graph.v))
                + " "
                + str(len(graph.e[0]))
                + " "
                + reform_target(target)
                + "\n"
            )
            f.write(" ".join(["0"] * len(graph.v)) + "\n")
            for edge in graph.e[0]:
                f.write(" ".join(map(lambda x: str(x), edge)) + "\n")

    return data


def reform_target(target):
    # retrun a string of target with witch type of structure in the graph
    result = []
    for i in range(len(target)):
        if target[i] != 0:
            result.append(str(i))
    s = " ".join(result)
    return s
