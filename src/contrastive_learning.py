import torch
import torch.nn.functional as F
from utils.logger import log
from utils.config import Config

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    log(f"   GPU: {torch.cuda.get_device_name(0)}")
    log(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    log("   Using CPU - consider installing CUDA for faster training")

def create_train_test_split(data, train_ratio=0.8):
    """Create train/test split for node classification"""
    num_nodes = data.x.size(0)
    num_train = int(num_nodes * train_ratio)
    
    # Create random permutation
    perm = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[perm[:num_train]] = True
    test_mask[perm[num_train:]] = True
    
    return train_mask, test_mask

def train_contrastive(model, data, optimizer, criterion, epochs=None):
    """Train the fraud detection model using supervised learning with GPU acceleration"""
    if epochs is None:
        epochs = Config.EPOCHS
        
    log(f"🚀 Starting fraud detection training for {epochs} epochs on {device}...")
    
    # Move model and data to GPU
    model = model.to(device)
    data = data.to(device)
    
    # Create train/test split
    train_mask, test_mask = create_train_test_split(data)
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)
    
    # Use actual labels from data
    labels = data.y.to(device)
    
    # Handle case with no fraud labels
    if labels.sum() == 0:
        log("⚠️ No fraud labels found, creating synthetic labels for training")
        labels = torch.randint(0, 2, (data.x.size(0),), device=device)
    
    best_loss = float('inf')
    
    # Enable mixed precision training for faster GPU training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        if len(train_mask.nonzero()) == 0:
            log("⚠️ No training samples available")
            break
        
        # Use mixed precision training for GPU acceleration
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # Forward pass
                out = model(data.x, data.edge_index)
                
                # Calculate loss only on training nodes
                train_out = out[train_mask]
                train_labels = labels[train_mask]
                loss = criterion(train_out, train_labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # CPU training (no mixed precision)
            out = model(data.x, data.edge_index)
            train_out = out[train_mask]
            train_labels = labels[train_mask]
            loss = criterion(train_out, train_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Log progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                test_out = out[test_mask]
                test_labels = labels[test_mask]
                
                if len(test_labels) > 0:
                    test_loss = criterion(test_out, test_labels)
                    train_acc = (train_out.argmax(dim=1) == train_labels).float().mean()
                    test_acc = (test_out.argmax(dim=1) == test_labels).float().mean()
                    
                    log(f"Epoch {epoch+1:3d}/{epochs} | "
                        f"Train Loss: {loss.item():.4f} | "
                        f"Test Loss: {test_loss.item():.4f} | "
                        f"Train Acc: {train_acc:.3f} | "
                        f"Test Acc: {test_acc:.3f}")
                else:
                    log(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {loss.item():.4f}")
        
        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
    
    log(f"✅ Training completed! Best loss: {best_loss:.4f}")
    return model
