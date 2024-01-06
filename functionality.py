import networkx as nx
import math

# 2.1.1
def graph_features(graph, graph_name):
    # Number of nodes in the graph
    nodes = graph.number_of_nodes()

    # Number of edges in the graph
    edges = graph.number_of_edges()

    # Graph density
    density = nx.density(graph)

    # Graph degree distribution
    # this is different based on the type of the graph
    if graph_name == "Citation Graph":
        # for citation graph we have to do both distinguish between in-degree and out-degree
        in_degrees = sorted([graph.in_degree(node) for node in graph.nodes()])

        in_degree_distro = {}
        for degree in in_degrees:
            if degree in in_degree_distro:
                in_degree_distro[degree] += 1
            else:
                in_degree_distro[degree] = 1
        
        out_degrees = sorted([graph.out_degree(node) for node in graph.nodes()])

        out_degree_distro = {}
        for degree in out_degrees:
            if degree in out_degree_distro:
                out_degree_distro[degree] += 1
            else:
                out_degree_distro[degree] = 1

    else:
        # collaboration graph is undirected

        degrees = sorted([graph.degree(node) for node in graph.nodes()])

        degree_distro = {}
        for degree in degrees:
            if degree in degree_distro:
                degree_distro[degree] += 1
            else:
                degree_distro[degree] = 1

    # Average degree of the graph
    avg_degree = 2 * edges / nodes

    # Calculating 95th percentile for graph hubs
    degrees = sorted([graph.degree(node) for node in graph.nodes()])
    percentile_95 = degrees[math.ceil(0.95 * len(degrees))]

    # Finding graph hubs
    graph_hubs = [node for node, degree in dict(graph.degree()).items() if degree > percentile_95]

    # Determining whether the graph is dense or sparse
    graph_type = "Dense" if density > 0.5 else "Sparse"

    # Returning the computed features

    if graph_name == "Citation Graph":
        # for citation graph

        return {
            "Graph Name": graph_name,
            "Number of Nodes": nodes,
            "Number of Edges": edges,
            "Graph Density": density,
            "Graph In-Degree Distribution": in_degree_distro,
            "Graph Out-Degree Distribution": out_degree_distro,
            "Average Degree": avg_degree,
            "Graph Hubs": graph_hubs,
            "Graph Type": graph_type
        }

    else:
        # for collaboration graph
        return {
            "Graph Name": graph_name,
            "Number of Nodes": nodes,
            "Number of Edges": edges,
            "Graph Density": density,
            "Graph Degree Distribution": degree_distro,
            "Average Degree": avg_degree,
            "Graph Hubs": graph_hubs,
            "Graph Type": graph_type
        }

# 2.1.2
def calculate_centralities(graph, graph_name, node):
    # Calculate centrality measures
    betweenness = nx.betweenness_centrality(graph)[node]
    pagerank = nx.pagerank(graph)[node]
    closeness = nx.closeness_centrality(graph)[node]
    degree = nx.degree_centrality(graph)[node]

    return {
        "Graph Name": graph_name,
        "Node": node,
        "Betweenness Centrality": betweenness,
        "PageRank": pagerank,
        "Closeness Centrality": closeness,
        "Degree Centrality": degree
    }

# 2.1.3
def shortest_ordered_walk_v2(graph, authors_a, a_1, a_n):
    # Check if a_1 and a_n are in the graph
    if a_1 not in graph.nodes() or a_n not in graph.nodes():
        return "One or more authors are not present in the graph."

    # Create a list to store the ordered nodes
    ordered_nodes = [a_1] + authors_a + [a_n]

    # Initialize an empty list to store the nodes in the shortest path
    shortest_path = []

    # Loop through the ordered_nodes list to find shortest paths between consecutive nodes
    for i in range(len(ordered_nodes) - 1):
        # Check if a path exists between the current node and the next node
        if not nx.has_path(graph, ordered_nodes[i], ordered_nodes[i + 1]):
            return "There is no such path."

        # Find the shortest path between current node (ordered_nodes[i]) and the next node (ordered_nodes[i + 1])
        path = nx.dijkstra_path(graph, ordered_nodes[i], ordered_nodes[i + 1])

        # Add the nodes from the path (excluding the last node) to the shortest_path list
        shortest_path.extend(path[:-1])

    # Append the final node a_n
    shortest_path.append(a_n)

    return shortest_path

#4
def min_edges_to_disconnect_v2(graph, author_a, author_b, top_authors):
    # Create a copy of the original graph
    modified_graph = graph.copy()

    # Create a subgraph with only the top authors
    top_authors_data = [author for author in top_authors if str(author) in graph.nodes()]
    subgraph = graph.subgraph(top_authors_data)

    # Remove edges in the subgraph from the temporary graph
    modified_graph.remove_edges_from(subgraph.edges())

    # Calculate edge connectivity between authors A and B
    edge_connectivity = nx.edge_connectivity(modified_graph, author_a, author_b)

    return edge_connectivity

#5
def extract_communities(graph, paper_1, paper_2, top_authors):
    # Extract the top authors' data
    top_authors_data = [author for author in top_authors if str(author) in graph.nodes()]

    # Create a copy of the original graph
    subgraph = graph.subgraph(top_authors_data)

    # Check if the subgraph has at least one node
    if len(subgraph) == 0:
        return "The subgraph has no nodes."

    # Detect communities using Louvain method
    communities = list(nx.algorithms.community.greedy_modularity_communities(subgraph))

    # Find the minimum number of edges to remove for communities to be seperated
    edges_to_remove = nx.algorithms.community.quality.modularity(subgraph, communities)

    return edges_to_remove