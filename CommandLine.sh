#!/bin/bash

# This bash script is the solution to the CLQ

# If no arguments are passed in input then the script will stops
# printing the help
args=($@)
if [ ${#args[@]} -eq 0 ]; then
	echo "The script must be launched with one of the following arguments:"
	echo "  1   'Is there any node that acts as an important connector between the different parts of the graph?'"
	echo "  2   'How does the degree of citation vary among the graph nodes?'"
	echo "  3   'What is the average length of the shortest path among nodes?'"
	exit
fi

filepath="adjacency.csv"

# Check if the adjacency matrix CSV exists, otherwise we launch
# the python script to build it
# Here we need pithon because we need to use networkx
if [ ! -f $filepath ]; then
    python3 scripts/savecsv.py
fi


# Then launch the script part related to the question specified as input
if [ $1 -eq 1 ]; then
	
	# We use awk to sum over each column and to write on the stdout a list of
	# node-id degree
	# Then we use sort to sort the numerical (-n) array in decreasing order (-r) by the degree field (-k 2) 
	# Then we create an array using readarray
	readarray -t table < <(awk -F ',' 'NR == 1 { for (i = 2; i <= NF; i++) nodeid[i] = $i; next } 
	{ for (i = 2; i <= NF; i++) degree[i] += $i } 
	END { for (i = 2; i <= NF; i++) print nodeid[i] " " degree[i] }' $filepath | sort -nr -k 2)
	
	# Evaluate the number of nodes, that is the length of the table 
	totalNodes=${#table[@]} 

	# Compute the id of the 95 percentile
	idxPerc95=$(echo "$totalNodes*(1-0.95)" | bc)
	idxPerc95=$(printf "%.0f" "$idxPerc95")
	
	# Print the number of Hubs
	echo $idxPerc95 "important Hubs"
	
	# Then print the list of all hubs with the relative degree
	# But here we print only the first 5 since the list is very long
	for ((i = 0; i<5; i++))
	do
		echo ${table[i]}
	done
	# We also print some dots to show that the results continue
	echo '...'
	
elif [ $1 -eq 2 ]; then
	
	# We use awk to read the adjacency matrix CSV and to write on the stdout 
	# with the following format:
	# degree-of-node number-of-nodes-with-that-degree
	# For example:
	# 0 1900
	# 1 500
	# 2 120
	# ....
	#
	# Then we redirect the stdout to gnuplot, that is a program that will plot the data  
	
	awk -F ',' 'NR > 1 { for (i = 2; i <= NF; i++) degree[i] += $i ;} 
		END {max = 0;
		for (i in degree) {
			if (degree[i] > max) {
				max = degree[i]
			}
		};
		for (i = 0; i <= max; i++){
			distro[i] = 0
		};
		for (j = 2; j<= length(degree);j++){  
			distro[degree[j]] += 1
		}; 
		for (i = 0; i <= max; i++){
			print i " " distro[i]
		}		
	}' $filepath | gnuplot -p -e "set title 'How the degree of citation vary among the graph nodes'; set xlabel 'Degree'; set ylabel 'Number of nodes'; plot '-' with linespoints , '';"
	
elif [ $1 -eq 3 ]; then
	
	# Launch the python script to evaluate the average shortest path using BFS  
	python3 scripts/clq.py
else
	echo "Argument not valid"
fi
