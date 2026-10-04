#!/usr/bin/env python3
"""
Agent harness for slow-inference that can poll to check if it's still running.
"""

import subprocess
import time
import os
import sys
import signal
import psutil

def is_process_running(pid):
    """Check if a process is still running."""
    try:
        p = psutil.Process(pid)
        return p.status() != psutil.ProcessStatus.STOPPED
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

def get_pid_of_slow_inference():
    """Get the PID of the slow-inference process if it's running."""
    # Common ways to find the process:
    # 1. Check if slow-inference binary is running
    try:
        result = subprocess.run(
            ["pgrep", "-f", "slow-inference"], 
            capture_output=True, 
            text=True,
            cwd="/home/reeds/slow-inference/engine"
        )
        if result.returncode == 0:
            pids = [int(pid) for pid in result.stdout.strip().splitlines()]
            return pids[0]  # return first PID
    except Exception as e:
        print(f"Error checking for slow-inference: {e}")
    
    # 2. Check for the specific command line pattern
    try:
        result = subprocess.run(
            ["pgrep", "-f", "slow-inference.*--model"], 
            capture_output=True, 
            text=True,
            cwd="/home/reeds/slow-inference/engine"
        )
        if result.returncode == 0:
            pids = [int(pid) for pid in result.stdout.strip().splitlines()]
            return pids[0]
    except Exception as e:
        print(f"Error checking for slow-inference: {e}")
    
    return None

def main():
    print("Slow-inference agent harness - polling for live processes")
    print("Press Ctrl+C to exit")
    
    last_pid = None
    check_interval = 5  # seconds
    
    while True:
        try:
            pid = get_pid_of_slow_inference()
            if pid:
                if not is_process_running(pid):
                    print(f"⚠️  Process {pid} appears to have stopped!")
                    sys.exit(1)
                else:
                    print(f"✅ Process {pid} is running")
            else:
                print("ℹ️  No slow-inference process found")
                
            # Check if the process is actually responding (not just running)
            # We could send a signal or try to read its status
            if last_pid and pid == last_pid:
                print("   👍 Still the same process")
            else:
                print(f"   🔄 PID changed: {last_pid} -> {pid}")
                
            last_pid = pid
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\n👋 Exiting agent harness")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(check_interval)

if __name__ == "__main__":
    main()
