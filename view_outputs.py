#!/usr/bin/env python3
"""
View CYBERFRAUDNET Results
"""
import os
import pandas as pd

def main():
    print("🛡️ CYBERFRAUDNET - Results Summary")
    print("=" * 40)
    
    # Check outputs
    if not os.path.exists("outputs"):
        print("❌ Run python main.py first")
        return
    
    # Show metrics
    metrics_file = "outputs/metrics_report.txt"
    if os.path.exists(metrics_file):
        print("📊 PERFORMANCE:")
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
            for line in lines[3:7]:  # Just the key metrics
                if ":" in line:
                    print(f"   {line.strip()}")
    
    # Show files
    print(f"\n📁 OUTPUTS:")
    for file in os.listdir("outputs"):
        if file.endswith(('.png', '.txt', '.pth')):
            size = os.path.getsize(f"outputs/{file}") / 1024
            print(f"   ✅ {file} ({size:.0f}KB)")
    
    # Show data
    if os.path.exists("data/processed/combined_fraud_data.csv"):
        df = pd.read_csv("data/processed/combined_fraud_data.csv")
        fraud_rate = df['is_fraud'].mean() * 100
        print(f"\n📈 DATASET: {len(df):,} samples, {fraud_rate:.1f}% fraud")
    
    print(f"\n🌐 DASHBOARD: python launch_flask_dashboard.py")

if __name__ == "__main__":
    main()