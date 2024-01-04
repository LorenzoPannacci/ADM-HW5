# ADM-HW5

## Team members
* Lorenzo Pannacci - 1948926
* ADD YOUR NAME HERE

This repository contains the submission of Group #11 of the fifth homework for the course "Algorithmic Methods of Data Mining", Academic year 2023–2024.
Here we provide the link for an easier access to the notebook: https://nbviewer.org/github/LorenzoPannacci/ADM-HW5/blob/main/main.ipynb

## Contents

* __`main.ipynb`__: the main notebook files. It contains all the answers and all the cells are already executed.

## Notes to be removed
### Loading the citation graph

```
import pickle
import networkx as nx

G_citation = pickle.load(open('graphs/citation-graph.pickle', 'rb'))
```

Then G_citation is a nx graph

### Loading the collaboration graph

```
import pickle
import networkx as nx

G_collaboration = pickle.load(open('graphs/collaboration-graph.pickle', 'rb'))
```

Then G_collaboration is a nx graph