from src.data_preprocessing import preprocess_data
from src.feature_extraction import extract_features
from src.graph_construction import build_graph
from src.model_gnn import TemporalGNN
from src.contrastive_learning import train_contrastive
from src.evaluation import evaluate_model
from src.explainability import explain_predictions
from utils.config import Config
from utils.logger import log

import torch

def main():
    log("🚀 Starting CYBERFRAUDNET Fraud Detection Pipeline")

    # Step 1: Load and preprocess all datasets
    data = preprocess_data()

    # Step 2: Feature extraction
    node_features = extract_features(data)

    # Step 3: Graph building
    graph_data = build_graph(data, node_features)

    # Step 4: Model initialization
    model = TemporalGNN(in_channels=graph_data.num_node_features,
                        hidden_channels=Config.HIDDEN_CHANNELS,
                        out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    criterion = torch.nn.CrossEntropyLoss()

    # Step 5: Contrastive training
    train_contrastive(model, graph_data, optimizer, criterion)

    # Step 6: Evaluation
    metrics = evaluate_model(model, graph_data)

    # Step 7: Save trained model
    import os
    os.makedirs('outputs', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/trained_model.pth')
    log("💾 Trained model saved to outputs/trained_model.pth")

    # Step 8: Explainability
    explain_predictions(model, graph_data)

    log("✅ CYBERFRAUDNET pipeline complete!")

if __name__ == "__main__":
    main()
