#!/usr/bin/env python3
"""
Simple training monitor script
Run this to check if training is still active
"""

import subprocess
import time
import os
from datetime import datetime

def check_training_status():
    """Check if Python training process is still running"""
    try:
        # Check for Python processes
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True, shell=True)
        
        if 'python.exe' in result.stdout:
            # Extract process info
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'python.exe' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[1]
                        memory = parts[4]
                        print(f"✅ Training is RUNNING")
                        print(f"   Process ID: {pid}")
                        print(f"   Memory Usage: {memory}")
                        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        return True
        
        print(f"❌ No Python training process found")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return False
        
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return False

def check_metrics_file():
    """Check if metrics file has been updated"""
    metrics_file = "data/processed/metrics.json"
    if os.path.exists(metrics_file):
        stat = os.stat(metrics_file)
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        print(f"📊 Metrics file last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        return True
    else:
        print(f"📊 No metrics file found")
        return False

if __name__ == "__main__":
    print("🔍 Training Status Monitor")
    print("=" * 40)
    
    # Check training process
    is_running = check_training_status()
    
    # Check metrics file
    check_metrics_file()
    
    print("=" * 40)
    
    if is_running:
        print("💡 Training is still in progress...")
        print("   The edge filtering operation takes time with 2.1M edges.")
    else:
        print("🎉 Training appears to be complete!")
        print("   Check the terminal for final results.") 