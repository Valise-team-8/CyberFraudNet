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
import gc
import os
import time

def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    log("🛡️ CYBERFRAUDNET - Advanced AI Fraud Detection Pipeline")
    
    # Check available hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"🚀 Using device: {device}")
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log(f"   GPU: {gpu_name}")
        log(f"   Memory: {gpu_memory:.1f} GB")
        
        # Optimize for RTX 2050 4GB VRAM
        if gpu_memory <= 4.5:
            torch.cuda.set_per_process_memory_fraction(0.8)
            log("   🔧 Optimized for 4GB VRAM")
    else:
        log("   Using CPU with optimizations")
    
    start_time = time.time()
    
    try:
        # Step 1: Load and preprocess data
        log("📊 Loading and preprocessing data...")
        data = preprocess_data()
        
        # Optimize data size for memory constraints
        original_size = len(data)
        if device.type == 'cuda' and gpu_memory <= 4.5 and len(data) > 75000:
            log("🔧 Using optimized data subset for GPU memory...")
            data = data.sample(n=75000, random_state=42).reset_index(drop=True)
            log(f"   Reduced from {original_size:,} to {len(data):,} samples")
        elif device.type == 'cpu' and len(data) > 50000:
            log("🔧 Using optimized data subset for CPU training...")
            data = data.sample(n=50000, random_state=42).reset_index(drop=True)
            log(f"   Reduced from {original_size:,} to {len(data):,} samples")
        
        clear_memory()

        # Step 2: Feature extraction
        log("🔍 Extracting features with memory optimization...")
        node_features = extract_features(data)
        clear_memory()

        # Step 3: Graph construction
        log("🕸️ Building graph...")
        graph_data = build_graph(data, node_features)
        clear_memory()

        # Step 4: Model initialization
        log("🧠 Initializing TemporalGNN model...")
        model = TemporalGNN(
            in_channels=graph_data.num_node_features,
            hidden_channels=Config.HIDDEN_CHANNELS,
            out_channels=2
        )
        
        # Move to device
        model = model.to(device)
        graph_data = graph_data.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR, weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss()
        
        log(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        clear_memory()

        # Step 5: Training with memory management
        log("🚀 Starting fraud detection training...")
        model = train_contrastive(model, graph_data, optimizer, criterion)
        clear_memory()

        # Step 6: Evaluation
        log("📊 Evaluating model performance...")
        metrics = evaluate_model(model, graph_data)

        # Step 7: Save model
        os.makedirs('outputs', exist_ok=True)
        torch.save(model.state_dict(), 'outputs/trained_model.pth')
        log("💾 Trained model saved to outputs/trained_model.pth")

        # Step 8: Explainability
        log("🔍 Generating model explanations...")
        explain_predictions(model, graph_data)

        # Training summary
        training_time = time.time() - start_time
        log("✅ CYBERFRAUDNET pipeline complete!")
        log(f"⏱️ Total training time: {training_time/60:.1f} minutes")
        log(f"🎯 Final Performance:")
        log(f"   Accuracy: {metrics.get('accuracy', 0):.1%}")
        log(f"   Precision: {metrics.get('precision', 0):.1%}")
        log(f"   Recall: {metrics.get('recall', 0):.1%}")
        log(f"   AUC: {metrics.get('auc', 0):.3f}")
        
        if device.type == 'cuda':
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            log(f"📊 Peak GPU Memory Used: {memory_used:.2f} GB")

    except RuntimeError as e:
        if "out of memory" in str(e):
            log("❌ GPU out of memory! Recommendations:")
            log("   1. Reduce BATCH_SIZE in utils/config.py")
            log("   2. Reduce HIDDEN_CHANNELS in utils/config.py")
            log("   3. Use smaller data subset")
            log("   4. Run: python train_fast_cpu.py for lighter training")
        else:
            log(f"❌ Training error: {e}")
    except Exception as e:
        log(f"❌ Unexpected error: {e}")
        log("💡 Try running: python train_fast_cpu.py for a simpler version")
    finally:
        clear_memory()

if __name__ == "__main__":
    main()
