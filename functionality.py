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

import heapq

def dijkstra_shortest_path(graph, start, end):
    # Initialize dictionaries to track distances and predecessors
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    predecessors = {node: None for node in graph}

    # Priority queue to keep track of nodes to explore
    queue = [(0, start)]  # Tuple of (distance, node)

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        # If we've already explored this node with a shorter distance, ignore it
        if current_distance > distances[current_node]:
            continue

        # If we've reached the destination node, construct the path and return it
        if current_node == end:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = predecessors[current_node]
            return path[::-1]  # Reverse the path to get from start to end

        # Explore neighbors of the current node
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            # If we've found a shorter path, update the information
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

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
        path = dijkstra_shortest_path(graph, ordered_nodes[i], ordered_nodes[i + 1])

        # Add nodes from the path (excluding the last node) to the shortest_path list
        shortest_path.extend(path[:-1] if path else [])

    # Append the final node a_n
    shortest_path.append(a_n)

    return shortest_path


#4



def build_adjacency_list(graph):
    adjacency_list = {}
    for node in graph.nodes():
        adjacency_list[node] = list(graph.neighbors(node))
    return adjacency_list

def dfs(adj_list, start, end, visited):
    visited.add(start)
    if start == end:
        return True
    for neighbor in adj_list[start]:
        if neighbor not in visited:
            if dfs(adj_list, neighbor, end, visited):
                return True
    return False

def min_edges_to_disconnect(graph, start_node, end_node, pass_through_nodes):
    adjacency_list = build_adjacency_list(graph)
    edges_removed = 0

    for edge in pass_through_nodes:
        if edge not in adjacency_list:
            return -1  # One or more nodes in pass_through_nodes are not present in the graph

    # Check if start_node and end_node are in the graph
    if start_node not in adjacency_list or end_node not in adjacency_list:
        return -1

    # Check if the graph is already disconnected
    if not dfs(adjacency_list, start_node, end_node, set()):
        return 0

    # Attempt to disconnect the graph by removing edges involving pass_through_nodes
    disconnected = False
    for edge in pass_through_nodes:
        if edge in adjacency_list[start_node]:
            adjacency_list[start_node].remove(edge)
            adjacency_list[edge].remove(start_node)
            edges_removed += 1

            # Check if the graph is disconnected after removing the edge
            if not dfs(adjacency_list, start_node, end_node, set()):
                disconnected = True
                break

    if disconnected:
        return edges_removed
    else:
        return -1  # Couldn't disconnect the graph





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
