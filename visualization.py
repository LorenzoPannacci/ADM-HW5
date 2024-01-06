import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def visualization_1(input_dict):
    fig, axs = plt.subplots(2, 2, height_ratios=[1, 3], figsize=(12, 8))

    # the first request is a table containing general information about the graph
    data_1 = [
        ["Number of Nodes", input_dict["Number of Nodes"]],
        ["Number of Edges", input_dict["Number of Edges"]],
        ["Graph Density", input_dict["Graph Density"]],
        ["Average Degree", input_dict["Average Degree"]],
        ["Network Type", input_dict["Graph Type"]]
    ]

    axs[0, 0].set_title('General Graph Information')
    axs[0, 0].axis('off')
    table_1 = axs[0, 0].table(cellText=data_1, colLabels=["Attribute", "Value"], loc='center', cellLoc='center')

    # make column and row names bold
    for (row, col), cell in table_1.get_celld().items():
        if row == 0 or col == 0:
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))

    # the second request is a table containing informations about the graph hubs

    if len(input_dict["Graph Hubs"]) <= 5:
        data_2 = [[hub] for hub in input_dict["Graph Hubs"]]
    else:
        data_2 = [[hub] for hub in input_dict["Graph Hubs"][:5]]

    axs[0, 1].set_title('Ordered list of top 5 graph hubs')
    axs[0, 1].axis('off')
    table_2 = axs[0, 1].table(cellText=data_2, colLabels=["Hub Name"], loc='center', cellLoc='center')

    # make column name bold
    for (row, col), cell in table_2.get_celld().items():
        if row == 0:
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))

    # now the content depends on the type of the graph
            
    if input_dict["Graph Name"] == "Citation Graph":
        # if citation graph plot the two degree distributions

        # in-degree distribution
        data_3 = input_dict["Graph In-Degree Distribution"]
        axs[1, 0].hist(data_3, bins=20, color='tab:blue')
        axs[1, 0].set_title("Graph In-Degree Distribution")

        # out-degree distribution
        data_4 = input_dict["Graph Out-Degree Distribution"]
        axs[1, 1].hist(data_4, bins=20, color='orange')
        axs[1, 1].set_title("Graph Out-Degree Distribution")

    else:
        # if colloboration graph plot the single degree distribution

        # degree distribution
        data_3 = input_dict["Graph Degree Distribution"]
        axs[1, 0].hist(data_3, bins=20, color='green')
        axs[1, 0].set_title("Graph Degree Distribution")

        # and hide second plot slot
        axs[1, 1].axis('off')


    if input_dict["Graph Name"] == "Citation Graph":
        plt.suptitle("Visualization of graph features for Citation Graph", fontsize=16)
    else:
        plt.suptitle("Visualization of graph features for Collaboration Graph", fontsize=16)

    plt.tight_layout()
    plt.show()

def visualization_2(input_dict):
    plt.figure(figsize=(6, 2))

    # create data from input dictionary:

    data = [
        ["Betweenness Centrality", input_dict["Betweenness Centrality"]],
        ["PageRank", input_dict["PageRank"]],
        ["Closeness Centrality", input_dict["Closeness Centrality"]],
        ["Degree Centrality", input_dict["Degree Centrality"]]
    ]

    table = plt.table(cellText=data, colLabels=["Measure", "Value"], loc='center', cellLoc='center')

    # make column and row names bold
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == 0:
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))

    plt.axis('off')
    plt.title("Contribution table for node " + str(input_dict["Node"]) + " for " + input_dict["Graph Name"])
    plt.show()