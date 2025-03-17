import json
import networkx as nx
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
import logging

def calculate_network_stats(G) -> dict:
    """Calculate basic network statistics."""
    degree_sequence = [d for n, d in G.degree()]
    return {
        'avg_degree': np.mean(degree_sequence),
        'density': nx.density(G),
        'degree_distribution': dict(Counter(degree_sequence)),
        'clustering_coefficient': nx.average_clustering(G),
        'global_clustering_coefficient': nx.transitivity(G),
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges()
    }

def create_network_data(correlation_matrix: pd.DataFrame, highlight_nodes: list = None) -> tuple:
    """
    Create network data from correlation matrix with optional highlighting.
    
    Args:
        correlation_matrix: Correlation matrix DataFrame
        highlight_nodes: List of node names to highlight (optional)
        
    Returns:
        tuple: (nodes_data, edges_data, network_stats)
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

    # Calculate network statistics
    network_stats = calculate_network_stats(G)
    
    # Log the nodes, edges, and stats
    logging.info(f"Nodes: {nodes}")
    logging.info(f"Edges: {edges}")
    logging.info(f"Network Stats: {network_stats}")

    return nodes, edges, network_stats

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
    nodes, edges, network_stats = create_network_data(correlation_matrix, highlight_nodes)
    
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Correlation Network Analysis</title>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style type="text/css">
            html, body {
                height: 100%;
                margin: 0;
                padding: 0;
                overflow: hidden;
                font-family: Arial, sans-serif;
            }
            .container {
                display: flex;
                width: 100%;
                height: 100%;
                overflow: hidden;
            }
            .network-container {
                flex: 60%;
                display: flex;
                flex-direction: column;
                min-height: 0; /* Critical for nested flex containers */
            }
            .stats-container {
                flex: 40%;
                padding: 20px;
                background-color: #f8f9fa;
                border-left: 1px solid #dee2e6;
                overflow-y: auto;
                height: 100%;
                box-sizing: border-box;
            }
            .explanation {
                padding: 10px;
                margin: 10px;
                background-color: #e9ecef;
                border-radius: 4px;
                flex-shrink: 0;
            }
            #controls {
                padding: 10px;
                background-color: #f8f9fa;
                border-bottom: 1px solid #dee2e6;
                flex-shrink: 0;
            }
            #mynetwork {
                flex: 1;
                min-height: 0; /* Critical for flex child */
                position: relative;
                border: 1px solid lightgray;
            }
            .stats-box {
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .stat-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
                margin-bottom: 20px;
            }
            .stat-item {
                background: #ffffff;
                padding: 10px;
                border-radius: 4px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .stat-value {
                font-size: 1.2em;
                font-weight: bold;
                color: #2c3e50;
            }
            .stat-label {
                color: #7f8c8d;
                font-size: 0.9em;
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
        <div class="container">
            <div class="network-container">
                <div class="explanation">
                    <p><strong>Node Size:</strong> Represents total correlation strength - larger nodes have stronger overall correlations.</p>
                    <p><strong>Edge Thickness:</strong> Shows correlation strength between variables.</p>
                    <p><strong>Highlighting:</strong> Red nodes with bold labels indicate specified variables.</p>
                    <p><strong>Controls:</strong> Adjust correlation threshold and node spacing using sliders.</p>
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
                        <input type="range" min="50" max="500" value="150" id="node-distance" style="width: 300px">
                        <span id="distance-value">150</span>
                    </div>
                </div>
                <div id="mynetwork"></div>
            </div>
            
            <div class="stats-container">
                <h2>Network Statistics</h2>
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="num-nodes">''' + str(network_stats['num_nodes']) + '''</div>
                        <div class="stat-label">Nodes</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="num-edges">''' + str(network_stats['num_edges']) + '''</div>
                        <div class="stat-label">Edges</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="avg-degree">''' + f"{network_stats['avg_degree']:.2f}" + '''</div>
                        <div class="stat-label">Average Degree</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="density">''' + f"{network_stats['density']:.3f}" + '''</div>
                        <div class="stat-label">Network Density</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="avg-clustering">''' + f"{network_stats['clustering_coefficient']:.3f}" + '''</div>
                        <div class="stat-label">Average Clustering Coefficient</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="global-clustering">''' + f"{network_stats['global_clustering_coefficient']:.3f}" + '''</div>
                        <div class="stat-label">Global Clustering Coefficient</div>
                    </div>
                </div>

                <div class="stats-box">
                    <h3>Degree Distribution</h3>
                    <canvas id="degreeDistChart"></canvas>
                </div>
            </div>
        </div>

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

            // Initialize degree distribution chart
            const degreeDistData = ''' + json.dumps(network_stats['degree_distribution']) + ''';
            const degreeCtx = document.getElementById('degreeDistChart').getContext('2d');
            const degreeChart = new Chart(degreeCtx, {
                type: 'bar',
                data: {
                    labels: Object.keys(degreeDistData),
                    datasets: [{
                        label: 'Number of Nodes',
                        data: Object.values(degreeDistData),
                        backgroundColor: 'rgba(54, 162, 235, 0.5)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Frequency'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Degree'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Node Degree Distribution'
                        }
                    }
                }
            });
        </script>
    </body>
    </html>
    '''

    # Save the HTML file
    output_path = Path(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Network visualization has been saved to {output_path.absolute()}")