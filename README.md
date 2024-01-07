# ADM-HW5

## Team members
* Lorenzo Pannacci - 1948926
* Rebecca Conti - 1896168
* Filippo Parlanti - 2136988
* Edo Fejzic

This repository contains the submission of Group #11 of the fifth homework for the course "Algorithmic Methods of Data Mining", Academic year 2023–2024.

Here we provide the link for an easier access to the notebook: https://nbviewer.org/github/LorenzoPannacci/ADM-HW5/blob/main/main.ipynb

## Contents

* __`main.ipynb`__: the main notebook files. It contains all the answers and all the cells are already executed
* __`functions/`__: a folder containing all the relevant functions used in the notebook divided into modules
  * `functionality.py`: a python script file that contains the funcionalities requested in the backend section of part 2
  * `visualization.py`: a python script file that contains the visualizations requested in the frontend section of part 2
  * `bonus.py`: a python script file that contains the functions used for the bonus question
* __`scripts/`__: a folder containing some scripts used in the main notebook
  * `bash_script.sh`: is used in data preprocessing (RQ1). In prticular, it splits the original database in smaller files. To be used it has to be moved in the same folder has the original .json file downloaded from Kaggle
  * `savecsv.py`: this script converts the citation graph file to a CSV adjacency matrix
  * `clq.py`: this script applies BFS to evaluate the average shortest path in the citation graph
* __`graphs/`__: is a folder containing the `citation graph` and the `collaboration graph` in the .pickle format, permitting to save a networkx graphs as files and load it later without have to compute them again
* __`images/`__:  is a folder containing tree images, one for each question of the CLQ
* __`CommandLine.sh`__: a bash script that solves the CLQ question