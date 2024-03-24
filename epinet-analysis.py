import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

# Data Processing Module
def load_data(nodes_file, edges_file):
    # Load nodes and edges from CSV files
    nodes = pd.read_csv(nodes_file)
    edges = pd.read_csv(edges_file)
    return nodes, edges

def construct_network(nodes, edges):
    G = nx.Graph()
    # Add nodes and edges with attributes
    for index, row in nodes.iterrows():
        G.add_node(row['ID'], **row.to_dict())
    for index, row in edges.iterrows():
        G.add_edge(row['SourceID'], row['TargetID'], **row.to_dict())
    return G

# Feature Engineering Module
selected_features = []  # Placeholder for predefined selected features
def generate_features(G):
    features = {}
    for node in G.nodes():
        node_data = G.nodes[node]
        feature_vector = [node_data.get(feature, 0) for feature in selected_features]  # if selected_features else []
        # Add network-based features
        feature_vector.extend([
            G.degree(node),
            nx.centrality.betweenness_centrality(G, normalized=True).get(node, 0),
            # Add other centrality measures as needed
        ])
        features[node] = feature_vector
    return features

# Machine Learning Module
def train_model(X, y, params):
    model = RandomForestClassifier(random_state=42)
    clf = GridSearchCV(model, params, cv=5)
    clf.fit(X, y)
    best_model = clf.best_estimator_
    return best_model, clf.best_params_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")
    print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

# Main Workflow
def main():
    nodes, edges = load_data('nodes.csv', 'edges.csv')
    G = construct_network(nodes, edges)
    features = generate_features(G)
    
    X = pd.DataFrame.from_dict(features, orient='index')
    y = nodes.set_index('ID')['Outcome']  # Assuming 'Outcome' is a column in nodes.csv indicating the target variable
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    params = {'n_estimators': [100, 200, 300]}
    model, best_params = train_model(X_train, y_train, params)
    
    print("Best Parameters:", best_params)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()