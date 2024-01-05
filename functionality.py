import networkx as nx
from collections import deque

#1
def graph_features(graph, graph_name):
    # Number of nodes in the graph
    nodes = graph.number_of_nodes()

    # Number of edges in the graph
    edges = graph.number_of_edges()

    # Graph density
    density = nx.density(graph)

    # Graph degree distribution
    degree_distro = dict(graph.degree())

    # Average degree of the graph
    avg_degree = sum(dict(graph.degree()).values()) / nodes

    # Calculating 95th percentile for graph hubs
    degree_values = sorted(list(dict(graph.degree()).values()))
    percentile_95 = degree_values[int(0.95 * len(degree_values))]

    # Finding graph hubs
    graph_hubs = [node for node, degree in dict(graph.degree()).items() if degree > percentile_95]

    # Determining whether the graph is dense or sparse
    graph_type = "Dense" if density >= 0.5 else "Sparse"

    # Returning the computed features
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

#2
def calculate_centralities(graph, graph_name, node):
    # Calculate centrality measures
    betweenness = nx.betweenness_centrality(graph)[node]
    pagerank = nx.pagerank(graph)[node]
    closeness = nx.closeness_centrality(graph)[node]
    degree = nx.degree_centrality(graph)[node]

    return {
        "Graph Name": graph_name,
        "Node": node,
        "Betweenness": betweenness,
        "PageRank": pagerank,
        "Closeness Centrality": closeness,
        "Degree Centrality": degree
    }

#3

def bfs_shortest_path(graph, start, end):
    # Dictionary to track visited nodes and their predecessors
    visited = {node: False for node in graph}
    visited[start] = True
    queue = deque([(start, [start])])  # Queue with the starting node and its partial path

    while queue:
        current_node, path = queue.popleft()

        # If we reach the destination node, return the path
        if current_node == end:
            return path

        # Explore neighbors of the current node
        for neighbor in graph[current_node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append((neighbor, path + [neighbor]))  # Add the node to the queue with the partial path

    return None  # If there's no path between start and end

def shortest_ordered_walk_v2(graph, authors_a, a_1, a_n):
    # Check if a_1 and a_n are in the graph
    if a_1 not in graph or a_n not in graph:
        return "One or more authors are not present in the graph."

    # Create a list to store the ordered nodes
    ordered_nodes = [a_1] + authors_a + [a_n]

    # Initialize an empty list to store nodes in the shortest path
    shortest_path = []

    # Loop through the ordered_nodes list to find shortest paths between consecutive nodes
    for i in range(len(ordered_nodes) - 1):
        # Check if a path exists between the current node and the next node
        if ordered_nodes[i + 1] not in graph[ordered_nodes[i]]:
            return "There is no such path."

        # Find the shortest path between current node (ordered_nodes[i]) and the next node (ordered_nodes[i + 1])
        path = bfs_shortest_path(graph, ordered_nodes[i], ordered_nodes[i + 1])

        # Add nodes from the path (excluding the last node) to the shortest_path list
        shortest_path.extend(path[:-1] if path else [])

    # Append the final node a_n
    shortest_path.append(a_n)

    return shortest_path


#4

class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = []

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, node1, node2):
        self.edges.append((node1, node2))
        self.edges.append((node2, node1))  # Assuming undirected graph

    def get_neighbors(self, node):
        neighbors = set()
        for edge in self.edges:
            if edge[0] == node:
                neighbors.add(edge[1])
        return neighbors


def subgraph_nodes(graph, top_authors):
    subgraph = Graph()
    subgraph.nodes = {node for node in graph.nodes if node in top_authors}
    subgraph.edges = [(node1, node2) for node1, node2 in graph.edges if node1 in top_authors and node2 in top_authors]
    return subgraph

def remove_edges(graph, edges):
    modified_graph = graph.copy()
    modified_graph.edges = [edge for edge in modified_graph.edges if edge not in edges]
    return modified_graph

def has_path(graph, start, end):
    visited = set()
    queue = [start]

    while queue:
        current_node = queue.pop(0)
        if current_node == end:
            return True  # There's a path between start and end

        visited.add(current_node)
        for edge in graph.edges:
            if edge[0] == current_node and edge[1] not in visited:
                queue.append(edge[1])

    return False  # There's no path between start and end

def min_edges_to_disconnect(graph, author_a, author_b, top_authors):
    # Create a subgraph with only the top authors
    subgraph = subgraph_nodes(graph, top_authors)

    edges_to_remove = subgraph.edges

    min_edges = 0
    while has_path(subgraph, author_a, author_b):
        if not edges_to_remove:
            return float('inf')  # No possible disconnection path found

        # Remove the "most important" edge
        edge = edges_to_remove.pop()
        subgraph = remove_edges(subgraph, [edge])
        min_edges += 1

    return min_edges





#5
class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}  # Storing edges as a dictionary of sets

    def add_node(self, node):
        self.nodes.add(node)
        self.edges[node] = set()  # Initialize edges for the node

    def add_edge(self, node1, node2):
        if node1 in self.edges and node2 in self.edges:
            self.edges[node1].add(node2)
            self.edges[node2].add(node1)  # Assuming undirected graph

def extract_communities(graph, paper_1, paper_2, top_authors):
    top_authors_data = [author for author in top_authors if author in graph.nodes]

    subgraph = Graph()
    for author in top_authors_data:
        subgraph.add_node(author)
        if isinstance(graph.edges, dict) and author in graph.edges:
            neighbors = graph.edges[author]
            if isinstance(neighbors, set):
                for neighbor in neighbors:
                    if neighbor in top_authors_data:
                        subgraph.add_edge(author, neighbor)

    if not subgraph.nodes:
        return "The subgraph has no nodes."

    communities = greedy_modularity_communities(subgraph)

    edges_to_remove = modularity(subgraph, communities)

    return edges_to_remove

def greedy_modularity_communities(graph):
    communities = []
    nodes = list(graph.nodes)
    while nodes:
        node = nodes.pop(0)
        neighbors = graph.edges[node]
        community = {node}
        previous_community_length = 0

        while len(community) > previous_community_length:
            previous_community_length = len(community)
            for neighbor in list(neighbors):
                if all(neigh in community or neigh not in graph.edges for neigh in graph.edges[neighbor]):
                    community.add(neighbor)
                    neighbors |= graph.edges[neighbor]

        communities.append(community)
        nodes = [n for n in nodes if n not in community]

    return communities

def modularity(graph, communities):
    modularity_score = 0
    for community in communities:
        for node in community:
            for neighbor in graph.edges[node]:
                if neighbor in community:
                    modularity_score += 1

    return modularity_score
