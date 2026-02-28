import pandas as pd
import numpy as np
import networkx as nx
import sys
import matplotlib.pyplot as plt

sys.path.append('../src')

variable_of_interest = "POLVIEWS"
first_year = 1972

# Initialise empty df
df = pd.DataFrame()

for i in range(1972, 2018, 2):
    G = nx.read_graphml(f"nbk/out/{i}_{i+3}.graphml")

    neighbours = [n for n in G.neighbors(variable_of_interest) if G[variable_of_interest][n]['weight'] != 0]
    neighbour_weight_dict = {n: G[variable_of_interest][n]['weight'] for n in neighbours}

    for neighbor, weight in neighbour_weight_dict.items():
        df.loc[i, neighbor] = abs(weight) if neighbor in df.columns else np.nan

    df = df.reindex(columns=df.columns.union(neighbours))

# Arrange the df columns in order of final size
df = df.reindex(df.iloc[-1].sort_values(ascending=False).index, axis=1)


# Set the figure size
fig_width = 22  # inches
fig_height = 8.5  # inches

# Increase font sizes
fontsize = 16

# Plot the area plot

# Save the plot
plot = df.plot.area(x=None, y=None, stacked=True, legend=0, figsize=(fig_width, fig_height), colormap="nipy_spectral", fontsize=fontsize)
plt.savefig(f'nbk/out/testfig1b_{variable_of_interest}.svg', bbox_inches='tight', dpi=300)



plot = df.plot.area(x=None, y=None, stacked=True, legend=1, figsize=(fig_width, fig_height), colormap="nipy_spectral", fontsize=fontsize)
plt.savefig(f'nbk/out/testfig1b_{variable_of_interest}_wLEGEND.svg', bbox_inches='tight', dpi=300)
print("Done!!")
