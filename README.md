
# Epilung EpiNet Analysis Project

## Overview
EpiNet Analysis is an AI-driven framework designed to analyze epidemiological data through network theory and machine learning. It aims to identify patterns, predict outcomes, and provide actionable insights based on large-scale health data sets. The project combines epidemiological modeling with advanced data analysis techniques to offer a comprehensive tool for researchers and public health officials.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Libraries
This project requires the following Python libraries:
- NetworkX
- NumPy
- scikit-learn
- matplotlib

### Steps
1. Clone the repository:
```bash
git clone https://github.com/heidihelena/epinet-analysis.git
```

2. Navigate to the project directory:
```bash
cd epinet-analysis
```

3. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

To use the EpiNet Analysis framework, follow these steps:

1. Prepare your epidemiological data according to the guidelines provided in the `data_format.md` file.
2. Adjust the epinet-analysis to your project and create main.py. Run the main script with Python:
```bash
python main.py
```
3. Follow the instructions in the script for inputting data and parameters.

For more detailed usage instructions, refer to the `docs` directory.

## Contributing

We welcome contributions from the community! If you're interested in contributing, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes with clear, descriptive messages.
4. Push the branch to your fork.
5. Submit a pull request to the main repository.

For more details, see the `CONTRIBUTING.md` file.

### Notes:
- **Feature Selection**: The placeholder for `selected_features` needs to be populated with actual feature names expected to be part of the nodes' attributes. These should match the column names in your nodes CSV file.
- **Outcome Variable**: The script now assumes there's an 'Outcome' column in your nodes CSV that represents the target variable for the ML model. Adjust the column name accordingly to match your data.
- **Visualization Implementation**: While the visualization function is still a placeholder, consider using `nx.draw` or libraries like `PyVis` for more advanced and interactive network visualizations.


To revise your code snippet for an automatic workflow that includes network visualization with `nx.draw` from NetworkX, I'll adjust your script to ensure it flows smoothly from loading data to generating a basic network visualization. This example assumes you have your network data prepared in two CSV files (`nodes.csv` and `edges.csv`), and you want to visualize the network as part of your analysis. 

Note: For this script to work as intended, ensure your `nodes.csv` includes at least the columns `ID` for node identifiers and any other attributes you wish to visualize or analyze. Similarly, `edges.csv` should have at least `SourceID` and `TargetID` to define the connections between nodes.

```python
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Data Processing Module
def load_data(nodes_file, edges_file):
    nodes = pd.read_csv(nodes_file)
    edges = pd.read_csv(edges_file)
    return nodes, edges

def construct_network(nodes, edges):
    G = nx.Graph()
    for index, row in nodes.iterrows():
        G.add_node(row['ID'], **row.to_dict())
    for index, row in edges.iterrows():
        G.add_edge(row['SourceID'], row['TargetID'], **row.to_dict())
    return G

# Visualization Module with nx.draw
def visualize_network(G):
    plt.figure(figsize=(12, 8))
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=50, edge_color='gray')
    plt.title('Network Visualization')
    plt.show()

# Main Workflow
def main():
    nodes, edges = load_data('nodes.csv', 'edges.csv')
    G = construct_network(nodes, edges)
    
    # Optional: Implement your feature engineering, ML model training, and evaluation here
    
    # For demonstration, we directly visualize the network
    visualize_network(G)

if __name__ == "__main__":
    main()
```

### Key Adjustments:
- **Visualization with `nx.draw`**: The `visualize_network` function uses `nx.draw` for a basic visualization. Adjustments like `node_color`, `node_size`, and `edge_color` are set for aesthetic purposes. You can customize these parameters further based on your needs.
- **Simplified Workflow**: The script is streamlined to focus on loading the data, constructing the network, and visualizing it. This setup assumes that feature engineering and machine learning components are either handled separately or can be inserted into the `main` workflow as needed.

### Additional Notes:
- For larger networks, or if you find the visualization cluttered, consider using `nx.spring_layout(G)` or similar layout algorithms for better node positioning.
- The script omits the machine learning components for brevity. Incorporate your model training and evaluation where the comment suggests, ensuring your features and labels are appropriately prepared from the network data.

When applying colors in nx.draw visualization, you can map them based on node attributes or metrics. For instance, if you have a metric that quantifies the centrality or importance of a node in the spread of asthma or COPD, you could map this metric to your color scale so that the most important nodes are highlighted with your primary colors.

A code example using NetworkX to apply epilung palette:

python
Copy code
def visualize_network(G, node_attribute):
    plt.figure(figsize=(12, 8))
    color_map = []

    # Define your color based on the node attribute or other criteria
    for node in G:
        if G.nodes[node][node_attribute] == 'condition_A':
            color_map.append('#87CEEB')  # Light blue for a certain condition
        elif G.nodes[node][node_attribute] == 'condition_B':
            color_map.append('#E8F196')  # Yellow for another condition
        else:
            color_map.append('#F8F0E6')  # Light grey for default nodes
    
    # Assuming 'pos' is a dictionary with node positions
    pos = nx.spring_layout(G)

    nx.draw(G, pos, node_color=color_map, with_labels=True, font_weight='bold', edge_color='#7C7873')
    plt.title('Network Visualization')
    plt.show()

## ... rest of your code to call visualize_network

This script provides a foundational workflow for network-based analysis, including a straightforward visualization step that can be further refined or expanded based on your project's specific requirements.


## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

For questions or further information, please contact the project maintainers:
- Dr. Heidi Andersén (heidi.andersen@tuni.fi)
- https://github.com/heidihelena
