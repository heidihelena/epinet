import networkx as nx
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

# Data Processing Module
def load_data(nodes_file, edges_file):
    try:
        nodes = pd.read_csv(nodes_file)
        edges = pd.read_csv(edges_file)
        print("Files loaded successfully.")
    except FileNotFoundError as e:
        raise RuntimeError(f"File not found: {e}")
    except pd.errors.ParserError as e:
        raise RuntimeError(f"Error parsing CSV: {e}")

    required_node_columns = {'ID', 'Outcome'}
    required_edge_columns = {'SourceID', 'TargetID'}

    if not required_node_columns.issubset(nodes.columns):
        raise ValueError(f"Nodes file must contain columns: {required_node_columns}")
    if not required_edge_columns.issubset(edges.columns):
        raise ValueError(f"Edges file must contain columns: {required_edge_columns}")

    nodes.fillna(0, inplace=True)
    edges.fillna(0, inplace=True)

    if nodes['Outcome'].dtype == 'object':
        nodes['Outcome'] = nodes['Outcome'].astype('category').cat.codes

    return nodes, edges

def construct_network(nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(nodes['ID'])
    G.add_edges_from(zip(edges['SourceID'], edges['TargetID']))
    return G

def generate_features(G):
    features = {}
    for node in G.nodes:
        features[node] = {
            "degree": G.degree[node],
            "clustering": nx.clustering(G, node),
        }
    return features

# Machine Learning Module
def train_model(X, y, params):
    print("Training model...")
    model = RandomForestClassifier(random_state=42)
    clf = GridSearchCV(
        model, params, cv=5, n_jobs=-1, scoring='f1'
    )
    clf.fit(X, y)
    best_model = clf.best_estimator_
    print("Model training complete.")
    print(f"Best Parameters: {clf.best_params_}")
    return best_model, clf.best_params_

def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Normalized)")
    plt.colorbar()
    plt.xticks([0, 1], labels=["Negative", "Positive"])
    plt.yticks([0, 1], labels=["Negative", "Positive"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title("Confusion Matrix (Raw)")
    plt.show()
    # Save confusion matrix plot
    plt.savefig("confusion_matrix.png")
    plt.show()
    print("Confusion matrix saved as 'confusion_matrix.png'.")

# Main Workflow
import argparse

def main():
    parser = argparse.ArgumentParser(description="Epinet Analysis Workflow")
    parser.add_argument('--nodes', type=str, default='nodes.csv', help='Path to nodes file')
    parser.add_argument('--edges', type=str, default='edges.csv', help='Path to edges file')
    args = parser.parse_args()

    nodes, edges = load_data(args.nodes, args.edges)

    # Ensure these functions are implemented
    G = construct_network(nodes, edges)
    features = generate_features(G)

    X = pd.DataFrame.from_dict(features, orient='index')
    y = nodes.set_index('ID')['Outcome']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    params = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15]}
    model, best_params = train_model(X_train, y_train, params)

    # Save the trained model
    dump(model, 'trained_model.joblib')
    print("Model saved as 'trained_model.joblib'")

    # Evaluate model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
   
