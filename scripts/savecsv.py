import networkx as nx 
import pickle

graphpath = "graphs/citation-graph.pickle"

G = pickle.load(open(graphpath, "rb"))
adj_df = nx.to_pandas_adjacency(G)
adj_df = adj_df.astype(int)
adj_df.to_csv('adjacency.csv', index=True, header=True)
