#!/usr/bin/env python3
"""
CYBERFRAUDNET Dashboard Launcher
"""
import subprocess
import sys
import os

def main():
    print("🛡️ CYBERFRAUDNET - AI Fraud Detection Dashboard")
    print("=" * 50)
    print("🚀 Launching futuristic dashboard...")
    print("📊 Features:")
    print("   • Real-time model performance metrics")
    print("   • Interactive data analysis")
    print("   • Live fraud prediction")
    print("   • System status monitoring")
    print("   • Futuristic cyberpunk UI")
    print()
    
    try:
        # Launch Streamlit app
        cmd = [sys.executable, "-m", "streamlit", "run", "demo_app/app.py", "--server.headless", "false"]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("💡 Make sure you have installed all requirements:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()