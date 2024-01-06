import networkx as nx
import math
from collections import deque
import heapq

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
def dijkstra_shortest_path(graph, start, end):
    # initialize dictionaries to track distances and predecessors
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    predecessors = {node: None for node in graph}

    # priority queue to keep track of nodes to explore
    queue = [(0, start)]  # Tuple of (distance, node)

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        # if we've already explored this node with a shorter distance, ignore it
        if current_distance > distances[current_node]:
            continue

        # if we've reached the destination node, construct the path and return it
        if current_node == end:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = predecessors[current_node]
            return path[::-1]  # reverse the path to get from start to end

        # explore neighbors of the current node
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight["weight"]

            # if we've found a shorter path, update the information
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    # if there's no path between start and end return None
    return None

def create_subgraph(graph, top_authors):
    # the first thing we have to do is to create the subgraph from the original graph
    # and the number top authors to keep

    # sort authors based on degree in descending order
    sorted_authors_by_degree = sorted(nx.degree_centrality(graph).items(), key=lambda x: x[1], reverse=True)

    # and keep only the top
    top_N_authors = sorted_authors_by_degree[:top_authors]

    # save the top authors in a list without the centrality degree value
    authors = [author[0] for author in top_N_authors]

    # create subgraph from graph
    graph = graph.subgraph(authors)

    return graph

def shortest_ordered_walk(graph, authors_a, a_1, a_n, top_authors):
    # create subgraph from graph
    graph = create_subgraph(graph, top_authors)

    # create a list to store the ordered nodes to visit
    ordered_nodes = [a_1] + authors_a + [a_n]

    # check if all elements we have to go trough are inside the graph
    for elem in ordered_nodes:
        if elem not in graph:
            return "One or more authors are not present in the graph."

    # initialize an empty list to store nodes in the shortest path
    shortest_path = []

    # loop through the ordered_nodes list to find shortest paths between consecutive nodes
    for i in range(len(ordered_nodes) - 1):

        # find the shortest path between current node (ordered_nodes[i]) and the next node (ordered_nodes[i + 1])
        path = dijkstra_shortest_path(graph, ordered_nodes[i], ordered_nodes[i + 1])

        # if there is no path return error message
        if path is None:
            return "There is no such path."

        # add nodes from the path to the shortest_path list
        shortest_path.extend(path[:-1])

    shortest_path.append(a_n)

    # now we have to build the paper informations about those paths
    traversed_papers = []
    id_list = []
    for i in range(len(shortest_path) - 1):
        start_node = shortest_path[i]
        end_node = shortest_path[i + 1]

        # create every edge we traversed
        edge = (start_node, end_node)

        # get all the papers involved
        for paper in graph.edges[edge]["papers"]:

            # do not insert duplicates
            if paper["id"] not in id_list:
                traversed_papers.append(paper)
                id_list.append(paper["id"])

    return shortest_path, traversed_papers

#2.1.4
# to work on this problem we create our own Graph class
class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = set()

# create graph
def init_graph(graph):
    subgraph = Graph()
    subgraph.nodes = {node for node in graph.nodes}
    subgraph.edges = {(node1, node2) for node1, node2 in graph.edges}
    return subgraph

# remove list of edges
def remove_edges(graph, edges):
    modified_graph = Graph()
    modified_graph.nodes = graph.nodes
    modified_graph.edges = {edge for edge in graph.edges if edge not in edges}
    return modified_graph

# find if graph has path from start to end
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
    # create a subgraph with only the top authors
    graph = create_subgraph(graph, top_authors)

    # convert to our type of graph
    subgraph = init_graph(graph)

    edges_to_remove = subgraph.edges

    min_edges = 0
    while has_path(subgraph, author_a, author_b):

        # Remove the "most important" edge
        edge = edges_to_remove.pop()
        subgraph = remove_edges(subgraph, [edge])
        min_edges += 1

    return min_edges

# 2.1.5
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