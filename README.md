# ADM-HW5

# Loading the citation graph

```
import pickle
import networkx as nx

G_citation = pickle.load(open('graphs/citation-graph.pickle', 'rb'))
```

Then G_citation is a nx graph

# Loading the collaboration graph

```
import pickle
import networkx as nx

G_collaboration = pickle.load(open('graphs/collaboration-graph.pickle', 'rb'))
```

Then G_collaboration is a nx graph