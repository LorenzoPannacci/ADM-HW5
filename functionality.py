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
# Function to find communities in the graph using the connected components algorithm
def find_communities(graph):
    def dfs(node, visited):
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, visited)

    visited = set()
    communities = []
    for node in graph:
        if node not in visited:
            component = []
            dfs(node, visited)
            communities.append(component)
    return communities

# Function to find the minimum number of edges to remove to form the communities
def min_edges_to_remove(graph, communities):
    edges_to_remove = 0
    for community in communities:
        subgraph = {node: graph[node] for node in community}
        edges_within_community = sum(len(subgraph[node]) for node in community) // 2
        edges_to_remove += edges_within_community-len(community) 
    return edges_to_remove

# Function to check if two papers belong to the same community
def same_community(communities, paper_1, paper_2):
    for community in communities:
        if paper_1 in community and paper_2 in community:
            return True
    return False
