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
    print("🚀 Launching cyberpunk dashboard...")
    print("📊 Features:")
    print("   • Real-time fraud detection with 90.7% accuracy")
    print("   • Interactive visualizations and analytics")
    print("   • File upload for live prediction")
    print("   • Cyberpunk UI with neon styling")
    print("   • System monitoring and performance metrics")
    print()
    
    try:
        # Change to flask_app directory
        os.chdir('flask_app')
        
        # Launch Flask app
        print("🌐 Starting Flask server...")
        print("📍 Dashboard: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop")
        print()
        
        cmd = [sys.executable, "app.py"]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()