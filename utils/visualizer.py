import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

def plot_graph(data):
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(6,5))
    nx.draw(G, node_size=40, node_color="skyblue", edge_color="gray")
    plt.title("CYBERFRAUDNET Transaction Graph")
    plt.show()

def plot_metrics(history):
    plt.figure(figsize=(6,4))
    plt.plot(history["loss"], label="Loss")
    plt.plot(history["accuracy"], label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Training Metrics")
    plt.show()
