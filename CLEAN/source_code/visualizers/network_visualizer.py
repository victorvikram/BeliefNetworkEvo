import json
import networkx as nx
from pathlib import Path
import pandas as pd
import numpy as np

def create_network_data(correlation_matrix: pd.DataFrame, highlight_nodes: list = None) -> tuple:
    """
    Create network data from correlation matrix with optional highlighting.
    
    Args:
        correlation_matrix: Correlation matrix DataFrame
        highlight_nodes: List of node names to highlight (optional)
        
    Returns:
        tuple: (nodes_data, edges_data)
    """
    # Set diagonal to zero to avoid self-links
    correlation_matrix = correlation_matrix.copy()
    np.fill_diagonal(correlation_matrix.values, 0)
    
    # Create network from correlation matrix
    G = nx.from_pandas_adjacency(abs(correlation_matrix))
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Calculate node values
    node_values = {}
    for node in G.nodes():
        edge_weights_sum = sum(abs(correlation_matrix.loc[node, :]))
        node_values[node] = edge_weights_sum

    # Prepare nodes for vis.js
    nodes = []
    for node in G.nodes():
        node_data = {
            'id': node,
            'label': node,
            'value': node_values[node],
            'title': f'Variable: {node}<br>Total Correlation Strength: {node_values[node]:.3f}'
        }
        # Add highlighting if specified
        if highlight_nodes and node in highlight_nodes:
            node_data.update({
                'color': '#FF0000',  # Red color
                'borderWidth': 3,
                'borderWidthSelected': 5,
                'font': {'size': 16, 'face': 'arial', 'bold': True}
            })
        nodes.append(node_data)

    # Prepare edges for vis.js
    edges = []
    for edge in G.edges():
        correlation = correlation_matrix.loc[edge[0], edge[1]]
        if correlation > 0:
            edges.append({
                'from': edge[0],
                'to': edge[1],
                'value': correlation,
                'title': f'Correlation: {correlation:.3f}'
            })

    return nodes, edges

def generate_html_visualization(
    correlation_matrix: pd.DataFrame, 
    output_path: str = 'correlation_network.html',
    highlight_nodes: list = None
) -> None:
    """
    Generate HTML visualization of the network with optional highlighting.
    
    Args:
        correlation_matrix: Input correlation matrix
        output_path: Path to save the HTML file (default: 'correlation_network.html')
        highlight_nodes: List of node names to highlight (optional)
    """
    nodes, edges = create_network_data(correlation_matrix, highlight_nodes)
    
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
            <p><strong>Highlighting:</strong> Specified nodes are shown in red with bold labels</p>
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
    output_path = Path(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Network visualization has been saved to {output_path.absolute()}")