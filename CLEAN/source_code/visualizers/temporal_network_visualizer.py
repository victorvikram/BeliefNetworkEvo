import sys
import os
from pathlib import Path
# Get the project root directory by going up 2 levels from this file

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from source_code.loaders.clean_raw_data import clean_datasets

import pandas as pd
import numpy as np
import json
from typing import Optional, List, Union, Dict, Any
import math 
import logging
import networkx as nx
from operator import itemgetter

from source_code.generators.corr_make_network import (
    calculate_correlation_matrix,
    CorrelationMethod,
    EdgeSuppressionMethod
)
from source_code.visualizers.network_visualizer import create_network_data

def generate_temporal_correlation_matrix(
    df: pd.DataFrame,
    variables_of_interest: Optional[List[str]] = None,
    time_window_length: int = 4,
    start_year: int = 2000,
    end_year: int = 2010,
    step_size: int = 2,
    method: Union[str, CorrelationMethod] = CorrelationMethod.SPEARMAN,
    partial: bool = False,
    edge_suppression: Union[str, EdgeSuppressionMethod] = EdgeSuppressionMethod.NONE,
    suppression_params: Optional[Dict[str, Any]] = None
) -> Dict[tuple, pd.DataFrame]:
    """
    Generate correlation matrices for temporal network analysis.
    
    Args:
        df: Input DataFrame containing the variables to analyze
        variables_of_interest: Optional list of variables to include in analysis
        time_window_length: Length of each time window in years
        start_year: Start year for analysis
        end_year: End year for analysis
        step_size: Number of years to step between windows
        method: Correlation method ('spearman' for non-linear or 'pearson' for linear)
        partial: Whether to compute partial correlations
        edge_suppression: Method to reduce weak edges
        suppression_params: Additional parameters for edge suppression
        
    Returns:
        Dict[tuple, DataFrame]: Dictionary mapping time windows to correlation matrices
    """
    correlation_matrices = {}
    
    print("Is this shit on")
    
    # Iterate through time windows
    for window_start in range(start_year, end_year - time_window_length + 1, step_size):
        window_end = window_start + time_window_length
        years_of_interest = list(range(window_start, window_end))

        corr_matrix = calculate_correlation_matrix(
            df=df,
            variables_of_interest=variables_of_interest,
            years_of_interest=years_of_interest,
            method=method,
            partial=partial,
            edge_suppression=edge_suppression,
            suppression_params=suppression_params
        )
        # Store matrix with time window tuple as key
        correlation_matrices[(window_start, window_end)] = corr_matrix
    
    return correlation_matrices

def generate_temporal_network_visualization(
    correlation_matrices: Dict[tuple, pd.DataFrame],
    output_path: str = 'temporal_network.html',
    highlight_nodes: Optional[List[str]] = None,
    time_window_length: int = 4,
    step_size: int = 2
) -> None:
    """
    Generate an interactive HTML visualization of temporal network evolution.
    
    Args:
        correlation_matrices: Dictionary of correlation matrices for each time window
        output_path: Path to save the HTML visualization
        highlight_nodes: Optional list of nodes to highlight in red
        time_window_length: Length of each time window in years
        step_size: Number of years between successive time windows
    """
    # Generate network data for each time window
    network_data = {}
    for time_window, corr_matrix in correlation_matrices.items():
        nodes, edges, stats = create_network_data_with_centrality(corr_matrix, highlight_nodes)
        
        # Add highlighting information to nodes
        if highlight_nodes:
            for node in nodes:
                if node['id'] in highlight_nodes:
                    node['color'] = {
                        'background': '#ff0000',
                        'border': '#cc0000'
                    }
                    node['font'] = {
                        'color': 'black',
                        'size': 16,
                        'face': 'arial',
                        'bold': True
                    }
        
        # Convert tuple key to string for JSON serialization
        window_key = f"{time_window[0]}-{time_window[1]}"
        network_data[window_key] = {
            'nodes': nodes,
            'edges': edges,
            'stats': stats
        }
    
    # Create time window labels for the slider
    time_windows = sorted(correlation_matrices.keys())
    time_labels = [f"{start}-{end}" for start, end in time_windows]
    # Convert time windows to string format for JSON
    time_windows_str = [f"{start}-{end}" for start, end in time_windows]

    # Pass temporal parameters to JavaScript
    temporal_params = {
        'start_year': time_windows[0][0],
        'end_year': time_windows[-1][1],
        'window_length': time_window_length,
        'step_size': step_size
    }

    # Add legend for highlighted nodes if any exist
    legend_html = ""
    if highlight_nodes:
        legend_html = '''
        <div class="legend">
            <h3>Highlighted Variables</h3>
            <ul>
            ''' + ''.join([f'<li><span class="highlight-dot"></span>{node}</li>' for node in highlight_nodes]) + '''
            </ul>
        </div>
        '''
    
    html_content = '''
    <!DOCTYPE html>
    <html>
        <head>
            <title>Temporal Network Analysis</title>
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
                    min-height: 0;
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
                .time-controls {
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-bottom: 1px solid #dee2e6;
                    display: flex;
                    align-items: center;
                    gap: 15px;
                }
                .time-slider {
                    flex-grow: 1;
                }
                .time-label {
                    font-size: 1.2em;
                    font-weight: bold;
                    min-width: 120px;
                }
                .play-button {
                    padding: 8px 16px;
                    background-color: #007bff;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }
                .play-button:hover {
                    background-color: #0056b3;
                }
                #mynetwork {
                    flex: 1;
                    min-height: 0;
                    position: relative;
                    border: 1px solid lightgray;
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
                .legend {
                    position: absolute;
                    top: 20px;
                    right: 20px;
                    background: rgba(255, 255, 255, 0.9);
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    z-index: 1000;
                }
                
                .legend h3 {
                    margin: 0 0 10px 0;
                    font-size: 14px;
                    color: #333;
                }
                
                .legend ul {
                    list-style: none;
                    padding: 0;
                    margin: 0;
                }
                
                .legend li {
                    display: flex;
                    align-items: center;
                    margin: 5px 0;
                    font-size: 12px;
                    color: #666;
                }
                
                .highlight-dot {
                    width: 10px;
                    height: 10px;
                    background: #ff0000;
                    border-radius: 50%;
                    margin-right: 8px;
                    display: inline-block;
                }
                .temporal-params {
                    margin-bottom: 20px;
                }
                .param-grid {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 15px;
                    margin-bottom: 20px;
                }
                .param-item {
                    background: #ffffff;
                    padding: 10px;
                    border-radius: 4px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    text-align: center;
                }
                .param-value {
                    font-size: 1.2em;
                    font-weight: bold;
                    color: #2c3e50;
                }
                .param-label {
                    color: #7f8c8d;
                    font-size: 0.9em;
                    margin-top: 5px;
                }
                .centrality-box {
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }
                .centrality-list {
                    margin-top: 10px;
                }
                .centrality-item {
                    display: flex;
                    justify-content: space-between;
                    padding: 8px;
                    border-bottom: 1px solid #eee;
                }
                .centrality-item:last-child {
                    border-bottom: none;
                }
                .centrality-node {
                    font-weight: bold;
                    color: #2c3e50;
                }
                .centrality-value {
                    color: #7f8c8d;
                }
                .centrality-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 10px;
                }
                .centrality-select {
                    padding: 5px;
                    border-radius: 4px;
                    border: 1px solid #dee2e6;
                    background-color: white;
                    font-size: 0.9em;
                    color: #2c3e50;
                    cursor: pointer;
                }
                .centrality-select:hover {
                    border-color: #adb5bd;
                }
                .centrality-description {
                    font-size: 0.9em;
                    color: #666;
                    margin-top: 5px;
                    margin-bottom: 10px;
                    font-style: italic;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="network-container">
                    ''' + legend_html + '''
                    <div class="time-controls">
                        <button id="playButton" class="play-button">Play</button>
                        <input type="range" id="timeSlider" class="time-slider" min="0" max="''' + str(len(time_windows)-1) + '''" value="0">
                        <span id="timeLabel" class="time-label">''' + time_labels[0] + '''</span>
                    </div>
                    <div class="time-controls">
                        <button id="physicsButton" class="play-button">Freeze Physics</button>
                    </div>
                    <div class="time-controls">
                        <label for="speedSlider" style="min-width: 180px;">Animation Speed (sec):</label>
                        <input type="range" id="speedSlider" class="time-slider" min="0.2" max="5" value="2" step="0.1">
                        <span id="speedLabel" class="time-label">2.0</span>
                    </div>
                    <div class="time-controls">
                        <label for="thresholdSlider" style="min-width: 180px;">Edge Threshold:</label>
                        <input type="range" id="thresholdSlider" class="time-slider" min="0" max="100" value="0" step="1">
                        <span id="thresholdLabel" class="time-label">0.00</span>
                    </div>
                    <div id="mynetwork"></div>
                </div>
                <div class="stats-container">
                    <div class="temporal-params">
                        <h2>About this data</h2>
                        <div class="param-grid">
                            <div class="param-item">
                                <div class="param-value">''' + str(temporal_params['start_year']) + '''</div>
                                <div class="param-label">Start Year</div>
                            </div>
                            <div class="param-item">
                                <div class="param-value">''' + str(temporal_params['end_year']) + '''</div>
                                <div class="param-label">End Year</div>
                            </div>
                            <div class="param-item">
                                <div class="param-value">''' + str(temporal_params['window_length']) + ''' years</div>
                                <div class="param-label">Time Window</div>
                            </div>
                            <div class="param-item">
                                <div class="param-value">''' + str(temporal_params['step_size']) + ''' years</div>
                                <div class="param-label">Step Size</div>
                            </div>
                        </div>
                    </div>
                    <h2>Network statistics</h2>
                    <div class="stat-grid">
                        <div class="stat-item">
                            <div class="stat-value" id="num-nodes"></div>
                            <div class="stat-label">Nodes</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="num-edges"></div>
                            <div class="stat-label">Edges</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="avg-degree"></div>
                            <div class="stat-label">Average Degree</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="density"></div>
                            <div class="stat-label">Network Density</div>
                        </div>
                    </div>
                    <div class="centrality-box">
                        <div class="centrality-header">
                            <h3>Top 5 Central Nodes</h3>
                            <select id="centralityType" class="centrality-select">
                                <option value="betweenness">Betweenness Centrality</option>
                                <option value="degree">Degree Centrality</option>
                                <option value="eigenvector">Eigenvector Centrality</option>
                                <option value="edge_weight">Edge Weight Centrality</option>
                            </select>
                        </div>
                        <div class="centrality-list" id="centrality-list">
                        </div>
                    </div>
                    <div class="stats-box">
                        <h3>Degree Distribution</h3>
                        <canvas id="degreeDistChart"></canvas>
                    </div>
                </div>
            </div>

            <script type="text/javascript">
                // Network data for all time windows
                const networkData = ''' + json.dumps(network_data) + ''';
                const timeWindows = ''' + json.dumps(time_windows_str) + ''';
                const timeLabels = ''' + json.dumps(time_labels) + ''';
                const highlightNodes = ''' + json.dumps(highlight_nodes if highlight_nodes else []) + ''';
                
                // Create network visualization
                const container = document.getElementById('mynetwork');
                
                // Initialize with first time window's data
                const initialTimeWindow = timeWindows[0];
                const initialData = networkData[initialTimeWindow];
                
                // Create DataSets with initial data
                const nodes = new vis.DataSet(initialData.nodes);
                const edges = new vis.DataSet(initialData.edges);

                // Configure network options with fixed positions for highlighted nodes
                const options = {
                    nodes: {
                        shape: 'dot',
                        scaling: {
                            min: 10,
                            max: 30,
                            label: { enabled: true, min: 14, max: 30 }
                        },
                        font: { size: 12, face: 'Arial' },
                        mass: 3
                    },
                    edges: {
                        width: 0.15,
                        color: { inherit: 'both' },
                        smooth: false,
                        length: 150
                    },
                    physics: {
                        enabled: true,
                        barnesHut: {
                            gravitationalConstant: -30000,
                            centralGravity: 0.5,
                            springLength: 150,
                            springConstant: 0.1,
                            damping: 0.3,
                            avoidOverlap: 0.5
                        },
                        stabilization: { 
                            enabled: true,
                            iterations: 200,
                            updateInterval: 25,
                            fit: true
                        },
                        timestep: 0.5
                    }
                };

                // Create network with options
                const network = new vis.Network(container, { nodes, edges }, options);

                // Keep track of current network state and threshold
                let currentNodes = new Set(initialData.nodes.map(node => node.id));
                let currentEdges = new Map(initialData.edges.map(edge => [`${edge.from}-${edge.to}`, edge]));
                let currentThreshold = 0;

                // Initialize degree distribution chart
                const degreeCtx = document.getElementById('degreeDistChart').getContext('2d');
                const degreeChart = new Chart(degreeCtx, {
                    type: 'bar',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Number of Nodes',
                            data: [],
                            backgroundColor: 'rgba(54, 162, 235, 0.5)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        animation: {
                            duration: 500 // Smoother transitions
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                min: 0,
                                max: 32,
                                ticks: {
                                    stepSize: 4
                                },
                                title: { 
                                    display: true, 
                                    text: 'Frequency' 
                                }
                            },
                            x: {
                                min: 0,
                                max: 16,
                                ticks: {
                                    stepSize: 2
                                },
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
                            },
                            legend: {
                                display: false
                            }
                        }
                    }
                });

                // Function to calculate degree distribution
                function calculateDegreeDistribution(nodeIds, edgeList) {
                    const degrees = {};
                    nodeIds.forEach(nodeId => {
                        degrees[nodeId] = 0;
                    });
                    edgeList.forEach(edge => {
                        degrees[edge.from] = (degrees[edge.from] || 0) + 1;
                        degrees[edge.to] = (degrees[edge.to] || 0) + 1;
                    });
                    
                    const degreeCounts = {};
                    Object.values(degrees).forEach(d => {
                        degreeCounts[d] = (degreeCounts[d] || 0) + 1;
                    });
                    return degreeCounts;
                }

                // Function to filter edges based on threshold
                function filterEdgesByThreshold(edges, threshold) {
                    return edges.filter(edge => Math.abs(edge.value) >= threshold);
                }

                // Centrality descriptions
                const centralityDescriptions = {
                    'betweenness': 'Measures how often a node acts as a bridge along the shortest path between two other nodes',
                    'degree': 'Measures the number of connections a node has to other nodes',
                    'eigenvector': 'Measures node importance based on the importance of its neighbors',
                    'edge_weight': 'Measures the total strength of all connections (sum of absolute edge weights) for each node'
                };

                // Centrality type selector
                const centralitySelect = document.getElementById('centralityType');
                let currentCentralityType = 'betweenness';
                let currentData = null;

                centralitySelect.addEventListener('change', (event) => {
                    currentCentralityType = event.target.value;
                    if (currentData && currentData.stats.centrality) {
                        updateCentralityList(currentData);
                    }
                });

                function updateCentralityList(data) {
                    const centralityList = document.getElementById('centrality-list');
                    if (!data || !data.stats.centrality || !data.stats.centrality[currentCentralityType]) {
                        centralityList.innerHTML = '<div class="centrality-description">No centrality data available</div>';
                        return;
                    }
                    
                    centralityList.innerHTML = `
                        <div class="centrality-description">${centralityDescriptions[currentCentralityType]}</div>
                        ${data.stats.centrality[currentCentralityType]
                            .map(item => `
                                <div class="centrality-item">
                                    <span class="centrality-node">${item.node}</span>
                                    <span class="centrality-value">${item.value}</span>
                                </div>
                            `)
                            .join('')}
                    `;
                }

                function updateVisualization(timeIndex, threshold = currentThreshold) {
                    // Store positions of current nodes
                    const positions = {};
                    nodes.get().forEach(node => {
                        positions[node.id] = network.getPosition(node.id);
                    });
                    
                    const timeWindow = timeWindows[timeIndex];
                    const data = networkData[timeWindow];
                    currentData = data;
                    
                    // Filter edges based on threshold
                    const filteredEdges = filterEdgesByThreshold(data.edges, threshold);
                    
                    // Update nodes and edges
                    nodes.clear();
                    edges.clear();
                    
                    // Add nodes with preserved positions but not fixed
                    const nodesWithPositions = data.nodes.map(node => {
                        if (positions[node.id]) {
                            return {
                                ...node,
                                x: positions[node.id].x,
                                y: positions[node.id].y
                            };
                        }
                        return node;
                    });
                    
                    nodes.add(nodesWithPositions);
                    edges.add(filteredEdges);
                    
                    // Update current state
                    currentNodes = new Set(data.nodes.map(node => node.id));
                    currentEdges = new Map(filteredEdges.map(edge => [`${edge.from}-${edge.to}`, edge]));
                    currentThreshold = threshold;

                    // Update statistics
                    document.getElementById('num-nodes').textContent = data.nodes.length;
                    document.getElementById('num-edges').textContent = filteredEdges.length;
                    
                    // Update average degree
                    const avgDegree = (2 * filteredEdges.length) / data.nodes.length || 0;
                    document.getElementById('avg-degree').textContent = avgDegree.toFixed(2);
                    
                    // Update density
                    const density = (2 * filteredEdges.length) / (data.nodes.length * (data.nodes.length - 1)) || 0;
                    document.getElementById('density').textContent = density.toFixed(3);
                    
                    // Update time label
                    document.getElementById('timeLabel').textContent = timeLabels[timeIndex];
                    
                    // Update degree distribution
                    const degreeCounts = calculateDegreeDistribution([...currentNodes], filteredEdges);
                    degreeChart.data.labels = Object.keys(degreeCounts);
                    degreeChart.data.datasets[0].data = Object.values(degreeCounts);
                    degreeChart.update('none');

                    // Update centrality list with current data
                    if (data.stats.centrality) {
                        updateCentralityList(data);
                    }

                    // Light stabilization
                    network.stabilize(1);
                }

                // Initialize sliders
                const timeSlider = document.getElementById('timeSlider');
                const thresholdSlider = document.getElementById('thresholdSlider');
                const thresholdLabel = document.getElementById('thresholdLabel');
                const speedSlider = document.getElementById('speedSlider');
                const speedLabel = document.getElementById('speedLabel');

                // Time slider event listener
                timeSlider.addEventListener('input', () => {
                    updateVisualization(parseInt(timeSlider.value), currentThreshold);
                });

                // Threshold slider event listener
                thresholdSlider.addEventListener('input', () => {
                    const threshold = parseFloat(thresholdSlider.value) / 100;
                    thresholdLabel.textContent = threshold.toFixed(2);
                    updateVisualization(parseInt(timeSlider.value), threshold);
                });

                // Speed slider event listener
                speedSlider.addEventListener('input', () => {
                    const speed = parseFloat(speedSlider.value);
                    speedLabel.textContent = speed.toFixed(1);
                    if (isPlaying) {
                        // Restart interval with new speed if currently playing
                        clearInterval(playInterval);
                        startPlayback();
                    }
                });

                // Play/pause functionality
                const playButton = document.getElementById('playButton');
                const physicsButton = document.getElementById('physicsButton');
                let isPlaying = false;
                let isPhysicsEnabled = true;
                let playInterval;

                function startPlayback() {
                    const speed = parseFloat(speedSlider.value) * 1000; // Convert to milliseconds
                    playInterval = setInterval(() => {
                        let currentValue = parseInt(timeSlider.value);
                        if (currentValue >= timeSlider.max) {
                            currentValue = 0;
                        } else {
                            currentValue++;
                        }
                        timeSlider.value = currentValue;
                        updateVisualization(currentValue, currentThreshold);
                    }, speed);
                }

                function togglePlay() {
                    if (isPlaying) {
                        clearInterval(playInterval);
                        playButton.textContent = 'Play';
                    } else {
                        startPlayback();
                        playButton.textContent = 'Pause';
                    }
                    isPlaying = !isPlaying;
                }

                playButton.addEventListener('click', togglePlay);

                // Physics toggle functionality
                physicsButton.addEventListener('click', () => {
                    isPhysicsEnabled = !isPhysicsEnabled;
                    network.setOptions({ physics: { enabled: isPhysicsEnabled } });
                    physicsButton.textContent = isPhysicsEnabled ? 'Freeze Physics' : 'Enable Physics';
                });

                // Initialize visualization with first time window
                updateVisualization(0);
            </script>
        </body>
    </html>
    '''

    # Save the HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Temporal network visualization has been saved to {output_path}")

def generate_temporal_html_visualization(df: pd.DataFrame,
                         nodes_to_highlight: List[str] = None,
                         time_window_length: int = 4,
                         start_year: int = 1972,
                         end_year: int = 2020,
                         step_size: int = 2,
                         method: Union[str, CorrelationMethod] = CorrelationMethod.PEARSON,
                         partial: bool = True,
                         edge_suppression: Union[str, EdgeSuppressionMethod] = EdgeSuppressionMethod.REGULARIZATION,
                         suppression_params: Dict = None,
                         output_path: str = 'delete_this_temporal_network.html') -> None:
    """
    Generate and plot a temporal network visualization from a cleaned GSS dataframe.
    
    This function takes a cleaned GSS dataframe and generates an interactive temporal network 
    visualization showing how correlations between variables evolve over time. The visualization
    is saved as an HTML file that can be viewed in a web browser.
    
    Args:
        df (pd.DataFrame): Cleaned GSS dataframe containing survey responses
        nodes_to_highlight (List[str], optional): List of node names to visually highlight. Defaults to None.
        time_window_length (int, optional): Length of rolling time window in years. Defaults to 10.
        start_year (int, optional): Start year for the temporal analysis. Defaults to 1972.
        end_year (int, optional): End year for the temporal analysis. Defaults to 2020.
        step_size (int, optional): Number of years between successive time windows. Defaults to 2.
        method (str, optional): Correlation method to use (e.g. Pearson, Spearman). Defaults to Pearson.
        partial (bool, optional): Whether to use partial correlations. Defaults to True.
        edge_suppression (str, optional): Method for suppressing weak edges. Defaults to regularization.
        suppression_params (Dict, optional): Parameters for edge suppression method. Defaults to None.
        output_path (str, optional): File path to save the HTML visualization. Defaults to 'delete_this_temporal_network.html'.
        
    Returns:
        A dictionary of correlation matrices for each time window.
    """
    # Set default regularization parameter if using partial correlations with regularization
    if partial and edge_suppression == EdgeSuppressionMethod.REGULARIZATION and suppression_params is None:
        suppression_params = {'regularization': 0.2}
        print("No regularization parameters provided, using default of 0.2")

    # Generate temporal correlation matrices for each time window    
    
    corr_matrices = generate_temporal_correlation_matrix(
        df,
        time_window_length=time_window_length,
        start_year=start_year,
        end_year=end_year,
        step_size=step_size,
        method=method,
        partial=partial,
        edge_suppression=edge_suppression,
        suppression_params=suppression_params
    )

    # Generate interactive visualization with highlighted nodes
    generate_temporal_network_visualization(
        corr_matrices,
        output_path=output_path,
        highlight_nodes=nodes_to_highlight
    )
    
    return corr_matrices

def create_network_data_with_centrality(corr_matrix: pd.DataFrame, highlight_nodes: Optional[List[str]] = None) -> tuple:
    """Create network data including multiple centrality metrics."""
    nodes, edges, stats = create_network_data(corr_matrix, highlight_nodes)
    
    # Create NetworkX graph from edges
    G = nx.Graph()
    for edge in edges:
        G.add_edge(edge['from'], edge['to'], weight=abs(edge['value']))
    
    # Calculate different centrality metrics
    betweenness = nx.betweenness_centrality(G, weight='weight')
    degree = nx.degree_centrality(G)
    eigenvector = nx.eigenvector_centrality_numpy(G, weight='weight')
    
    # Calculate edge weight centrality (sum of absolute edge weights)
    edge_weight = {}
    for node in G.nodes():
        edge_weight[node] = sum(abs(G[node][neighbor]['weight']) for neighbor in G[node])
    
    # Store all centrality metrics
    stats['centrality'] = {
        'betweenness': [{'node': node, 'value': round(value, 3)} 
                       for node, value in sorted(betweenness.items(), key=itemgetter(1), reverse=True)[:5]],
        'degree': [{'node': node, 'value': round(value, 3)} 
                  for node, value in sorted(degree.items(), key=itemgetter(1), reverse=True)[:5]],
        'eigenvector': [{'node': node, 'value': round(value, 3)} 
                       for node, value in sorted(eigenvector.items(), key=itemgetter(1), reverse=True)[:5]],
        'edge_weight': [{'node': node, 'value': round(value, 3)} 
                       for node, value in sorted(edge_weight.items(), key=itemgetter(1), reverse=True)[:5]]
    }
    
    return nodes, edges, stats

def test_temporal_correlation_matrix():
    """Test the temporal correlation matrix generation and visualization."""
    cleaned_df = clean_datasets()

    # Example nodes to highlight (these are belief variables we want to track)
    nodes_to_highlight = [
        'POLVIEWS',  # Political views
        'TRUST',     # Trust in people
        'GRASS',     # Marijuana legalization
        'CAPPUN',    # Capital punishment
        'GUNLAW'     # Gun control
    ]

    # Generate temporal correlation matrices
    corr_matrices = generate_temporal_correlation_matrix(
        cleaned_df,
        time_window_length=4,
        start_year=2000,
        end_year=2010,
        step_size=2,
        method=CorrelationMethod.PEARSON,
        partial=True,
        edge_suppression=EdgeSuppressionMethod.REGULARIZATION,
        suppression_params={'regularization': 0.2}
    )

    # Generate visualization with highlighted nodes
    generate_temporal_network_visualization(
        corr_matrices,
        output_path='temporal_network.html',
        highlight_nodes=nodes_to_highlight
    )


if __name__ == "__main__":
    test_temporal_correlation_matrix()
