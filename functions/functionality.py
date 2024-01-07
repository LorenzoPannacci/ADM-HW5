import networkx as nx
import math
from collections import deque
import heapq
import random

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

# 2.1.4
def dijkstra(G, start_node, end_node, nodes_to_consider):
    # Initialize node weights
    distances = {node: float('inf') for node in G}
    distances[start_node] = 0

    # Initialize predecessors
    predecessors = {node: None for node in G}

    unvisited_nodes = set(nodes_to_consider)  # Consider only nodes provided

    while unvisited_nodes:
        current_node = min(unvisited_nodes, key=lambda node: distances[node])

        if distances[current_node] == float('inf'):
            break

        unvisited_nodes.remove(current_node)

        for neighbor, edge_weight in G[current_node].items():
            total_weight = distances[current_node] + edge_weight['weight']

            if total_weight < distances[neighbor]:
                distances[neighbor] = total_weight
                predecessors[neighbor] = current_node

    # Construct the shortest path
    shortest_path = []
    node = end_node
    while node is not None:
        shortest_path.insert(0, node)
        node = predecessors[node]

    return shortest_path

def min_edges_to_disconnect(G, start_node, end_node, nodes_to_consider):
    # Find the shortest path between start_node and end_node using only nodes_to_consider
    shortest_path = dijkstra(G, start_node, end_node, nodes_to_consider)

    # Count the number of connections in the path
    num_edges_to_remove = len(shortest_path) - 1 if shortest_path else 0

    return num_edges_to_remove

#2.1.5
def BFS_visit(graph, start_node):
    # start data structures
    visited = {}
    queue = deque()

    # insert starting node
    queue.append(start_node)
    visited[start_node] = True

    # cycle trough the queue
    while len(queue) != 0:
        # get current node
        current_node = queue.popleft()

        # find all the neighbors of the current node
        for node in graph.neighbors(current_node):
                if node not in visited.keys():
                    # insert them in the queue if haven't already visited
                    queue.append(node)
                    visited[node] = True

    return visited

def weakly_connected_components(graph):
    # since we are searching for weakly connected components we convert the graph
    # into a undirected graph
    graph = nx.Graph(graph)

    # we do BFS visits to find connected components and continue until we have
    # no nodes left
    communities = []
    while len(graph.nodes) != 0:
        # select the starting node for the BFS at random
        random_node = random.choice(list(graph.nodes()))

        # get the list of visited nodes
        community = set(BFS_visit(graph, random_node).keys())

        # insert the new community into the community list
        communities.append(community)
        
        # remove the nodes in the community from the graph
        graph.remove_nodes_from(community)
    
    return communities

def girvan_newman(graph, n_communities):
    # calculate the current communities as the weakly connected components
    communities = weakly_connected_components(graph)

    # continue until the desidered number of communities is reached
    n_edges_removed = 0
    while len(communities) < n_communities:

        # calculate the betweenness centrality of every edge in the graph
        edge_betweenness = nx.edge_betweenness_centrality(graph)

        # find the edge to remove as the one with the highest score
        # in case there is more than one remove all the nodes with the same score
        max_score = max(edge_betweenness.values())

        # get and remove all the edges with the max score
        for edge, value in edge_betweenness.items():
            if value == max_score:
                graph.remove_edge(*edge)
                n_edges_removed += 1

        # adjurn the communities list
        communities = list(weakly_connected_components(graph))

    # when reached the desidered number of communities end the algorithm
    return n_edges_removed, communities

def functionality_5(graph, n_nodes, n_communities, paper_1, paper_2):
    # create subgraph
    graph = nx.DiGraph(create_subgraph(graph, n_nodes))

    # call girvan newman algorithm
    n_edges_removed, communities = girvan_newman(graph, n_communities)

    # search if the two papers are in the same community
    same_community = False
    for community in communities:
        if paper_1 in community and paper_2 in community:
            same_community = True
    
    return graph, n_edges_removed, communities, same_community
