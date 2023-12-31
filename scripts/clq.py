import networkx as nx

# Function to find the average shortest path
def total_shortest_path(graph):
    total_shortest_distances = 0
    num_pairs = 0

    # Iterate through all pairs of nodes
    for source in graph.nodes:
        # Perform BFS from the source node to all other nodes
        paths = nx.single_source_shortest_path_length(graph, source)
        for target, distance in paths.items():
            if source != target:
                total_shortest_distances += distance
                num_pairs += 1

    return f"{total_shortest_distances},{num_pairs}"

