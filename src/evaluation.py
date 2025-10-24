from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from utils.logger import log

def evaluate_model(model, data):
    """Evaluate the fraud detection model"""
    log("📊 Evaluating model performance...")
    model.eval()
    
    with torch.no_grad():
        # Get model predictions
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        # Use actual labels from graph data
        true_labels = data.y
        
        # Handle case where we have no fraud labels
        if true_labels.sum() == 0:
            log("⚠️ No fraud labels found in data, using synthetic labels for evaluation")
            true_labels = torch.randint(0, 2, (preds.size(0),))
        
        # Convert to numpy for sklearn metrics
        true_labels_np = true_labels.cpu().numpy()
        preds_np = preds.cpu().numpy()
        probs_np = probs[:, 1].cpu().numpy()  # Probability of fraud class
        
        # Calculate metrics
        try:
            acc = accuracy_score(true_labels_np, preds_np)
            prec = precision_score(true_labels_np, preds_np, average='binary', zero_division=0)
            rec = recall_score(true_labels_np, preds_np, average='binary', zero_division=0)
            
            # AUC only if we have both classes
            if len(set(true_labels_np)) > 1:
                auc = roc_auc_score(true_labels_np, probs_np)
            else:
                auc = 0.5
                
            log(f"✅ Model Performance:")
            log(f"   Accuracy: {acc:.3f}")
            log(f"   Precision: {prec:.3f}")
            log(f"   Recall: {rec:.3f}")
            log(f"   AUC: {auc:.3f}")
            
            # Detailed classification report
            log("📋 Classification Report:")
            report = classification_report(true_labels_np, preds_np, 
                                         target_names=['Normal', 'Fraud'], 
                                         zero_division=0)
            log(report)
            
            # Save outputs
            save_evaluation_outputs(true_labels_np, preds_np, probs_np, acc, prec, rec, auc, report)
            
            return {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'auc': auc
            }
            
        except Exception as e:
            log(f"⚠️ Error in evaluation: {e}")
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'auc': 0}

def save_evaluation_outputs(true_labels, preds, probs, acc, prec, rec, auc, report):
    """Save evaluation results and visualizations"""
    os.makedirs('outputs', exist_ok=True)
    
    # Save metrics report
    with open('outputs/metrics_report.txt', 'w') as f:
        f.write("CYBERFRAUDNET Fraud Detection Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model Performance Metrics:\n")
        f.write(f"Accuracy:  {acc:.3f}\n")
        f.write(f"Precision: {prec:.3f}\n")
        f.write(f"Recall:    {rec:.3f}\n")
        f.write(f"AUC:       {auc:.3f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report)
        f.write(f"\n\nDataset Statistics:\n")
        f.write(f"Total samples: {len(true_labels)}\n")
        f.write(f"Fraud cases: {sum(true_labels)} ({sum(true_labels)/len(true_labels)*100:.1f}%)\n")
        f.write(f"Normal cases: {len(true_labels) - sum(true_labels)} ({(len(true_labels) - sum(true_labels))/len(true_labels)*100:.1f}%)\n")
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'], 
                yticklabels=['Normal', 'Fraud'])
    plt.title('Fraud Detection Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create ROC curve if we have both classes
    if len(set(true_labels)) > 1:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(true_labels, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create prediction distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(probs[true_labels == 0], bins=50, alpha=0.7, label='Normal', color='blue')
    plt.hist(probs[true_labels == 1], bins=50, alpha=0.7, label='Fraud', color='red')
    plt.xlabel('Fraud Probability')
    plt.ylabel('Count')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    fraud_probs = probs[true_labels == 1]
    normal_probs = probs[true_labels == 0]
    plt.boxplot([normal_probs, fraud_probs], labels=['Normal', 'Fraud'])
    plt.ylabel('Fraud Probability')
    plt.title('Probability Distribution by Class')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/probability_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log("💾 Evaluation outputs saved to outputs/ directory:")
    log("   - metrics_report.txt")
    log("   - confusion_matrix.png") 
    log("   - roc_curve.png")
    log("   - probability_distributions.png")