#!/usr/bin/env python3
"""
Simple script to view CYBERFRAUDNET outputs
"""
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def main():
    print("🛡️ CYBERFRAUDNET - Fraud Detection Results")
    print("=" * 50)
    
    # Check if outputs exist
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        print("❌ No outputs directory found. Run main.py first.")
        return
    
    # Display metrics report
    metrics_file = os.path.join(outputs_dir, "metrics_report.txt")
    if os.path.exists(metrics_file):
        print("\n📊 MODEL PERFORMANCE METRICS:")
        print("-" * 30)
        with open(metrics_file, 'r') as f:
            print(f.read())
    else:
        print("❌ Metrics report not found.")
    
    # List all output files
    print("\n📁 GENERATED OUTPUT FILES:")
    print("-" * 30)
    for file in os.listdir(outputs_dir):
        if os.path.isfile(os.path.join(outputs_dir, file)):
            file_path = os.path.join(outputs_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"✅ {file} ({file_size:,} bytes)")
    
    # Display processed data info
    processed_data = "data/processed/combined_fraud_data.csv"
    if os.path.exists(processed_data):
        print(f"\n📈 PROCESSED DATASET:")
        print("-" * 30)
        df = pd.read_csv(processed_data)
        print(f"✅ Dataset shape: {df.shape}")
        print(f"✅ Fraud cases: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.1f}%)")
        print(f"✅ Columns: {list(df.columns)}")
    else:
        print("❌ Processed data not found.")
    
    print(f"\n🎯 HOW TO VIEW VISUALIZATIONS:")
    print("-" * 30)
    print("1. Open the PNG files in outputs/ directory with any image viewer")
    print("2. Run: python -m streamlit run demo_app/app.py (for interactive dashboard)")
    print("3. Files are located at:")
    for file in ["confusion_matrix.png", "roc_curve.png", "probability_distributions.png"]:
        file_path = os.path.join(outputs_dir, file)
        if os.path.exists(file_path):
            print(f"   📊 {os.path.abspath(file_path)}")

if __name__ == "__main__":
    main()