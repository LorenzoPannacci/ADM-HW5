import pandas as pd
import numpy as np
import networkx as nx
from pyspark import SparkContext, SparkConf
from tqdm import tqdm

def process_rows(row):
    if isinstance(row, str):
        # remove brackets and split
        row = row.strip('[]').split(',')

        # convert elements to int
        row = [int(i) for i in row]

    elif pd.isna(row):
        # assign an empty list for NaN
        row = []

    return row

def prepare_graph(n_papers = 100000):
    # load from csv
    df_small = pd.read_csv("small_dataset.csv", index_col='id')

    # since moving it into a csv converted the list into a string we have to take it back to a string
    df_small['references'] = df_small['references'].apply(process_rows)

    df_small['n_citation'] = df_small['n_citation'].astype(int)
    df_small = df_small.sort_values(by=['n_citation'], ascending=False).nlargest(n_papers, 'n_citation')

    # to create the graph we have to create the list of nodes and the list of edges
    # the list of nodes is immediate as is the list indices of our dataframe

    # we have to create a list containing all the edges, each edge wil be a tuple (from, to)

    exploded = df_small.references.explode()

    # converting it to a list of tuples with the index
    edges_list = list(zip(exploded.index, exploded))

    # removing the edges that have NaN values
    edges_list = list(filter(lambda x: not np.isnan(x[1]), edges_list))
    edges_list = np.array(edges_list)

    # removing the edges that point to nodes outside the subset we created
    indices = np.isin(edges_list[:, 1], df_small.index.values)
    edges_list = edges_list[indices]

    # initialize an empty directed graph 
    G_citation = nx.DiGraph()

    # adding the node to the graph
    G_citation.add_nodes_from(df_small.index.values)

    # adding the edges to the graph
    G_citation.add_edges_from(edges_list)

    print(f"In the citation graph there are {len(G_citation.edges)} edges and {len(G_citation.nodes)} nodes")

    return G_citation

def page_rank(G_citation, iterations = 20, beta = 0.85):

    # initialize SparkContext
    sc = SparkContext("local", "PageRank")

    # load edges as a list of tuples containing the two IDs of the nodes forming the edge
    # the first element is the "source" of the edge and the second is the destination
    edges = sc.parallelize(G_citation.edges())

    # also load the nodes as a list
    # we will need the number of nodes to calculate scores later
    nodes = sc.parallelize(G_citation.nodes())
    n_nodes = len(G_citation.nodes())
        
    # convert the previous list into a list of tuples where the first element is a node
    # and the second is a list of all the node you can reach from the starting one
    # (the first element will act as "key" in the MapReduce sense)
    links = edges.groupByKey().mapValues(list).cache()

    # initialize PageRank scores
    # since the stationary distribution is unique the starting value brings no difference in the
    # outcome of the function, we decided to start with a uniform distribution for each node
    page_ranks = nodes.map(lambda node: (node, 1 / n_nodes))

    # we will also need empty contributions for every node to avoid the lose of keys during iterations
    empty_contributions = nodes.map(lambda node: (node, 0))
        
    # perform PageRank iterations
    for _ in tqdm(range(iterations)):
        
        # for each iteration we have to update the probability we end up in that certain node by
        # applying the transition matrix to the current state
        # since explicitly do matrix calculations would be computationally really expensive
        # instead of doing that we can do the following

        # join for each node its score and the nodes it can reach
        contributions = links.join(page_ranks)

        # count is need to avoid the lazy behavior of Spark
        contributions.count()

        page_ranks.unpersist(blocking=True)

        # divide the current node score by the number of outgoing nodes and creating tuples
        # in this way we are computing the probability that each node give to each other node it can reach
        # with this scores we create tuples of the outgoing nodes and the score they get
        # note that each node will have a tuple for each edge that has it as its destination
        contributions = contributions.flatMap(lambda x: [(destination, x[1][1] / len(x[1][0])) for destination in x[1][0]])

        # union with empty contributions to avoid the lost of nodes without edges pointing at them
        contributions = contributions.union(empty_contributions)

        # update the scores
        # merge the previously obtained scores based on the destination (the "key")
        page_ranks = contributions.reduceByKey(lambda x, y: x + y)

        contributions.unpersist(blocking=True)

        # scale the previous score by the probability of using a real path and add
        # a constant factor to every node linked to the random teleport probability
        page_ranks = page_ranks.mapValues(lambda x: beta * x + (1 - beta) / n_nodes)

    # collect final PageRank values
    final_page_ranks = page_ranks.collect()

    # covert into dictionary
    my_pagerank = {key: value for key, value in final_page_ranks}

    # stop SparkContext
    sc.stop()

    return my_pagerank