# region Imports
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import json
from pathlib import Path
# endregion

# region Load Data
# Load in the cleaned dataset
df_cleaned_1, meta = pd.read_pickle('datasets/cached_data/gss_cleaned_1.pkl')
# endregion

# Filter the data between a time period
start_year = 2020
end_year = 2021
df_cleaned_1 = df_cleaned_1[(df_cleaned_1['YEAR'] >= start_year) & (df_cleaned_1['YEAR'] <= end_year)]

# region Overlap Matrix Calculation and Visualization
# Create a matrix showing percentage of overlapping non-null values
total_rows = len(df_cleaned_1)
overlap_matrix = pd.DataFrame(index=df_cleaned_1.columns, columns=df_cleaned_1.columns)
# Only iterate through upper triangle
for i, col1 in enumerate(df_cleaned_1.columns):
    for j, col2 in enumerate(df_cleaned_1.columns[i:], i):
        overlap = df_cleaned_1[[col1, col2]].dropna().shape[0]
        percentage = (overlap / total_rows) * 100  # Convert to percentage
        overlap_matrix.loc[col1, col2] = percentage
        if col1 != col2:  # Don't copy diagonal values
            overlap_matrix.loc[col2, col1] = percentage

overlap_matrix = overlap_matrix.astype(float) # Convert to numeric type explicitly

plt.figure(figsize=(12, 10))
sns.heatmap(overlap_matrix, 
            cmap='YlOrRd',
            xticklabels=True,
            yticklabels=True,
            fmt='.1f')  # Show one decimal place
plt.title('Percentage of Overlapping Values Between Variables (%)')
plt.xticks(rotation=45, ha='right', fontsize=6)
plt.yticks(rotation=0, fontsize=6)
plt.tight_layout()
plt.show()
# endregion

# region Correlation Matrix Calculation and Visualization
# Create a correlation matrix
correlation_matrix = df_cleaned_1.corr(method='spearman')

# Square every value in the matrix
correlation_matrix = correlation_matrix ** 2

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='YlOrRd')
plt.title('Spearman Correlation Matrix')
plt.tight_layout()
plt.show()
# endregion

# region Network Visualization Setup
# Create network data
G = nx.from_pandas_adjacency(abs(correlation_matrix))
pos = nx.spring_layout(G, k=1, iterations=50)

# Calculate node values based on sum of edge weights
node_values = {}
for node in G.nodes():
    # Sum all correlation values (edge weights) connected to this node
    edge_weights_sum = sum(abs(correlation_matrix.loc[node, :]))
    node_values[node] = edge_weights_sum

# Prepare nodes and edges for vis.js
nodes = []
for node in G.nodes():
    nodes.append({
        'id': node,
        'label': node,
        'value': node_values[node],  # Node size based on sum of correlations
        'title': f'Variable: {node}<br>Total Correlation Strength: {node_values[node]:.3f}'  # Updated tooltip
    })

edges = []
for edge in G.edges():
    correlation = correlation_matrix.loc[edge[0], edge[1]]
    if correlation > 0:  # Only add edges with correlation
        edges.append({
            'from': edge[0],
            'to': edge[1],
            'value': correlation,  # Edge width based on correlation
            'title': f'Correlation: {correlation:.3f}'  # Hover tooltip
        })

# Create the HTML content
html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>Correlation Network</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        #mynetwork {
            width: 100%;
            height: 800px;
            border: 1px solid lightgray;
        }
        #container {
            width: 100%;
            height: 100%;
        }
        #controls {
            padding: 10px;
            background-color: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }
        .explanation {
            padding: 10px;
            margin: 10px 0;
            background-color: #e9ecef;
            border-radius: 4px;
        }
        .slider-container {
            margin: 10px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        label {
            display: inline-block;
            width: 200px;
        }
        .threshold-inputs {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        input[type="number"] {
            width: 80px;
            padding: 5px;
        }
    </style>
</head>
<body>
    <div class="explanation">
        <h3>Network Visualization Guide</h3>
        <p><strong>Node Size:</strong> The size of each node (variable) represents the sum of all correlation strengths connected to that variable. 
           Larger nodes indicate variables that have stronger overall correlations with other variables.</p>
        <p><strong>Edge Thickness:</strong> The thickness of connections between nodes represents the strength of correlation between those variables.</p>
        <p><strong>Controls:</strong> Use the correlation threshold slider to filter weak correlations, and the node distance slider to adjust the network layout.</p>
    </div>
    <div id="controls">
        <div class="slider-container">
            <label for="threshold">Correlation Threshold:</label>
            <div class="threshold-inputs">
                <input type="range" min="0" max="100" value="5" id="threshold" style="width: 300px">
                <input type="number" id="threshold-number" min="0" max="1" step="0.01" value="0.05" style="width: 80px">
            </div>
        </div>
        <div class="slider-container">
            <label for="node-distance">Node Distance:</label>
            <input type="range" min="50" max="300" value="150" id="node-distance" style="width: 300px">
            <span id="distance-value">150</span>
        </div>
    </div>
    <div id="mynetwork"></div>

    <script type="text/javascript">
        // Network data
        const nodes_data = ''' + json.dumps(nodes) + ''';
        const edges_data = ''' + json.dumps(edges) + ''';
        
        // Create a DataSet for nodes and edges
        const nodes = new vis.DataSet(nodes_data);
        const edges = new vis.DataSet(edges_data);
        
        // Create network data
        const data = {
            nodes: nodes,
            edges: edges
        };
        
        // Network options
        const options = {
            nodes: {
                shape: 'dot',
                scaling: {
                    min: 10,
                    max: 30,
                    label: {
                        enabled: true,
                        min: 14,
                        max: 30,
                        maxVisible: 30,
                        drawThreshold: 5
                    }
                },
                font: {
                    size: 12,
                    face: 'Arial'
                }
            },
            edges: {
                width: 0.15,
                color: { inherit: 'both' },
                smooth: {
                    type: 'continuous'
                },
                scaling: {
                    min: 1,
                    max: 5
                }
            },
            physics: {
                barnesHut: {
                    gravitationalConstant: -80000,
                    centralGravity: 0.3,
                    springLength: 150,
                    springConstant: 0.04,
                    damping: 0.09,
                    avoidOverlap: 0.1
                },
                maxVelocity: 50,
                minVelocity: 0.1,
                solver: 'barnesHut',
                stabilization: {
                    enabled: true,
                    iterations: 1000,
                    updateInterval: 100
                },
                timestep: 0.5,
                adaptiveTimestep: true
            }
        };
        
        // Create network
        const container = document.getElementById('mynetwork');
        const network = new vis.Network(container, data, options);
        
        // Initialize network with default threshold of 0.05
        updateNetwork(0.05);
        
        // Threshold controls
        const thresholdSlider = document.getElementById('threshold');
        const thresholdNumber = document.getElementById('threshold-number');
        
        function updateNetwork(threshold) {
            // Filter edges based on threshold
            const filteredEdges = edges_data.filter(edge => edge.value >= threshold);
            edges.clear();
            edges.add(filteredEdges);
            
            // Update node values based on visible connections
            const nodeConnections = {};
            filteredEdges.forEach(edge => {
                nodeConnections[edge.from] = (nodeConnections[edge.from] || 0) + 1;
                nodeConnections[edge.to] = (nodeConnections[edge.to] || 0) + 1;
            });
            
            const updatedNodes = nodes_data.map(node => ({
                ...node,
                value: nodeConnections[node.id] || 0,
                title: `Variable: ${node.id}<br>Connections: ${nodeConnections[node.id] || 0}`
            }));
            
            nodes.clear();
            nodes.add(updatedNodes);
        }
        
        // Sync slider and number input
        thresholdSlider.oninput = function() {
            const threshold = this.value / 100;
            thresholdNumber.value = threshold.toFixed(2);
            updateNetwork(threshold);
        };
        
        thresholdNumber.oninput = function() {
            const threshold = parseFloat(this.value);
            if (!isNaN(threshold) && threshold >= 0 && threshold <= 1) {
                thresholdSlider.value = threshold * 100;
                updateNetwork(threshold);
            }
        };
        
        // Node distance slider functionality
        const distanceSlider = document.getElementById('node-distance');
        const distanceValue = document.getElementById('distance-value');
        
        distanceSlider.oninput = function() {
            const distance = parseInt(this.value);
            distanceValue.textContent = distance;
            
            network.setOptions({
                physics: {
                    barnesHut: {
                        springLength: distance
                    }
                }
            });
        };
    </script>
</body>
</html>
'''

# Save the HTML file
output_path = Path('correlation_network.html')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Network visualization has been saved to {output_path.absolute()}")
# endregion

# region shortest path betweenness centrality calculation
import numpy as np
from collections import deque

import numpy as np
import heapq

def betweenness_centrality_weighted(adj_matrix):
    """
    Calculate the betweenness centrality for each node in a weighted graph using a
    weighted version of Brandes' algorithm.

    Parameters:
        adj_matrix (numpy.ndarray): A square numpy array representing the weighted 
                                    adjacency matrix of the graph. An entry of 0 
                                    indicates no edge.

    Returns:
        bc (numpy.ndarray): A 1D numpy array where bc[i] is the betweenness centrality of node i.
    """
    n = adj_matrix.shape[0]
    bc = np.zeros(n, dtype=float)  # Betweenness centrality for each node

    for s in range(n):
        # Initialization
        S = []  # Stack of nodes in order of non-decreasing distance from s.
        P = [[] for _ in range(n)]  # Predecessors on shortest paths from s.
        sigma = np.zeros(n, dtype=float)  # Number of shortest paths from s to each node.
        sigma[s] = 1.0
        dist = np.full(n, np.inf)  # Distance from s to each node.
        dist[s] = 0.0

        # Priority queue for Dijkstra's algorithm: (distance, node)
        queue = []
        heapq.heappush(queue, (0.0, s))

        while queue:
            d, v = heapq.heappop(queue)
            # If we found a better path before, skip this one.
            if d > dist[v]:
                continue
            S.append(v)
            # Iterate over all possible neighbors w of v
            for w in range(n):
                weight = adj_matrix[v, w]
                if weight != 0:
                    new_dist = dist[v] + weight
                    if new_dist < dist[w]:
                        dist[w] = new_dist
                        sigma[w] = sigma[v]
                        heapq.heappush(queue, (new_dist, w))
                        P[w] = [v]
                    # If we found an equally short path, update sigma and predecessors.
                    elif np.isclose(new_dist, dist[w]):
                        sigma[w] += sigma[v]
                        P[w].append(v)

        # Dependency accumulation: back-propagate the dependencies.
        delta = np.zeros(n, dtype=float)
        while S:
            w = S.pop()
            for v in P[w]:
                # The fraction of shortest paths through v that pass through w.
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            # Do not include the source in its own betweenness.
            if w != s:
                bc[w] += delta[w]

    # For undirected graphs, the betweenness centrality is halved.
    bc = bc / 2.0
    return bc


bc = betweenness_centrality_weighted(correlation_matrix.to_numpy())
# Sort nodes by betweenness centrality in descending order
sorted_nodes = sorted(enumerate(bc), key=lambda x: x[1], reverse=True)

# region Centrality Measures Comparison
# Calculate degree and edge weight sums
degrees = np.sum(correlation_matrix != 0, axis=0) - 1  # Subtract 1 to exclude self-loops
edge_weight_sums = np.sum(correlation_matrix, axis=0) - 1  # Subtract 1 to exclude self-loops

# Create DataFrames for each measure
betweenness_df = pd.DataFrame({
    'Variable': correlation_matrix.index,
    'Betweenness Centrality': bc
}).sort_values('Betweenness Centrality', ascending=False).head(10)

degree_df = pd.DataFrame({
    'Variable': correlation_matrix.index,
    'Degree': degrees
}).sort_values('Degree', ascending=False).head(10)

strength_df = pd.DataFrame({
    'Variable': correlation_matrix.index,
    'Total Correlation': edge_weight_sums
}).sort_values('Total Correlation', ascending=False).head(10)

# Format the values
betweenness_df['Betweenness Centrality'] = betweenness_df['Betweenness Centrality'].map('{:.4f}'.format)
degree_df['Degree'] = degree_df['Degree'].map('{:.0f}'.format)
strength_df['Total Correlation'] = strength_df['Total Correlation'].map('{:.4f}'.format)

# Add rank numbers
betweenness_df.index = range(1, 11)
degree_df.index = range(1, 11)
strength_df.index = range(1, 11)


# Print dataset information
print("\nDataset Information:")
print(f"Time period: {start_year}-{end_year}")
print(f"Number of respondents: {len(df_cleaned_1):,}")
print(f"Number of belief variables: {len(correlation_matrix.columns):,}")
print(f"Average non-null responses per belief: {df_cleaned_1.notna().sum().mean():.1f}")
print("\n" + "="*80 + "\n")


# Combine the dataframes side by side
print("\nTop 10 Variables by Different Centrality Measures:\n")
combined_rankings = pd.concat([betweenness_df, degree_df, strength_df], axis=1)

# Print with nice formatting
print(combined_rankings.to_string(justify='left'))
# endregion
