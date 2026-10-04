#!/usr/bin/env python3
"""
Agent Harness for Slow-Inference Engine
Provides polling mechanism to check if slow-inference is still alive/running
and handles long-running inference tasks with timeouts and health checks.
"""

import subprocess
import time
import os
import sys
import signal
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import psutil


class SlowInferenceHarness:
    def __init__(self, model_path: str, engine_path: Optional[str] = None):
        self.model_path = model_path
        self.engine_path = engine_path or "/home/reeds/slow-inference/engine/target/release/slow-inference"
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.last_health_check = 0
        self.health_check_interval = 5  # seconds
        
        # Validate paths exist
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"Engine not found at {self.engine_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        print(f"✓ Harness initialized")
        print(f"  Engine: {self.engine_path}")
        print(f"  Model: {self.model_path}")

    def start(self, prompt: str = "", max_tokens: int = 50, 
              json_output: bool = False) -> subprocess.Popen:
        """Start slow-inference process with given parameters."""
        if self.is_running:
            print("⚠️  Process already running, stopping first...")
            self.stop()
            
        cmd = [
            self.engine_path,
            "--model", self.model_path,
        ]
        
        if prompt:
            cmd.extend(["--prompt", prompt])
        if max_tokens != 50:  # default is often 8, but let's be explicit
            cmd.extend(["--tokens", str(max_tokens)])
        if json_output:
            cmd.append("--json")
            
        print(f"🚀 Starting slow-inference: {' '.join(cmd)}")
        
        # Start process with pipes for communication
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            cwd="/home/reeds/slow-inference/engine"
        )
        
        self.is_running = True
        self.last_health_check = time.time()
        print(f"✅ Started process PID: {self.process.pid}")
        return self.process

    def is_alive(self) -> bool:
        """Check if the process is still running."""
        if not self.process:
            return False
            
        poll_result = self.process.poll()
        if poll_result is None:
            # Process is still running
            self.last_health_check = time.time()
            return True
        else:
            # Process has exited
            print(f"💀 Process exited with code: {poll_result}")
            self.is_running = False
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the process."""
        health_info = {
            "timestamp": time.time(),
            "alive": False,
            "pid": None,
            "exit_code": None,
            "uptime": None,
            "memory_mb": None,
            "cpu_percent": None
        }
        
        if not self.process:
            health_info["error"] = "No process started"
            return health_info
            
        poll_result = self.process.poll()
        if poll_result is None:
            # Process is running
            health_info["alive"] = True
            health_info["pid"] = self.process.pid
            
            try:
                ps_process = psutil.Process(self.process.pid)
                health_info["memory_mb"] = round(ps_process.memory_info().rss / 1024 / 1024, 2)
                health_info["cpu_percent"] = ps_process.cpu_percent(interval=0.1)
                health_info["uptime"] = time.time() - ps_process.create_time()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                health_info["error"] = "Cannot access process info"
        else:
            health_info["alive"] = False
            health_info["exit_code"] = poll_result
            self.is_running = False
            
        return health_info

    def wait_for_completion(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for the process to complete, with optional timeout."""
        if not self.process:
            return {"error": "No process started"}
            
        try:
            stdout, stderr = self.process.communicate(timeout=timeout)
            self.is_running = False
            return {
                "stdout": stdout,
                "stderr": stderr,
                "returncode": self.process.returncode,
                "timed_out": False
            }
        except subprocess.TimeoutExpired:
            # Kill the process on timeout
            self.stop()
            return {
                "error": f"Process timed out after {timeout} seconds",
                "timed_out": True,
                "pid": self.process.pid if self.process else None
            }

    def stop(self):
        """Stop the process gracefully, then forcefully if needed."""
        if not self.process:
            return
            
        print(f"🛑 Stopping process PID: {self.process.pid}")
        
        # Try graceful termination first
        self.process.terminate()
        
        try:
            # Wait for graceful shutdown
            self.process.wait(timeout=5)
            print("✅ Process terminated gracefully")
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't stop
            print("⚠️  Process didn't stop gracefully, forcing kill...")
            self.process.kill()
            self.process.wait()
            print("💀 Process force killed")
            
        self.is_running = False
        self.process = None

    def monitor(self, check_interval: float = 5.0, duration: Optional[float] = None):
        """Monitor the process and report health status."""
        start_time = time.time()
        print(f"📊 Monitoring process (check every {check_interval}s)")
        if duration:
            print(f"   Will stop after {duration}s")
            
        try:
            while self.is_running:
                # Check if we've exceeded duration
                if duration and (time.time() - start_time) > duration:
                    print(f"⏰ Monitoring duration exceeded ({duration}s)")
                    break
                    
                health = self.health_check()
                status = "🟢 Alive" if health["alive"] else "🔴 Dead"
                
                info_parts = [status]
                if health["pid"]:
                    info_parts.append(f"PID:{health['pid']}")
                if health["memory_mb"] is not None:
                    info_parts.append(f"RAM:{health['memory_mb']}MB")
                if health["cpu_percent"] is not None:
                    info_parts.append(f"CPU:{health['cpu_percent']:.1f}%")
                if health["uptime"] is not None:
                    info_parts.append(f"UP:{health['uptime']:.0f}s")
                    
                print(" | ".join(info_parts))
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n👋 Monitoring interrupted by user")
        finally:
            if self.is_running:
                self.stop()


def example_usage():
    """Example of how to use the harness."""
    print("=" * 60)
    print("Slow-Inference Agent Harness Example")
    print("=" * 60)
    
    # Initialize harness
    harness = SlowInferenceHarness(
        model_path="/home/reeds/slow-inference/models/Tiny-Moe.Q4_K_M.gguf"
    )
    
    try:
        # Start a generation task
        print("\n📝 Starting generation task...")
        process = harness.start(
            prompt="The future of AI is",
            max_tokens=10,
            json_output=False
        )
        
        # Monitor the process
        print("\n📊 Monitoring process health...")
        harness.monitor(check_interval=3.0, duration=20)  # Monitor for 20 seconds
        
        # Or wait for completion with timeout
        # print("\n⏳ Waiting for completion (10s timeout)...")
        # result = harness.wait_for_completion(timeout=10)
        # print(f"Result: {json.dumps(result, indent=2)[:500]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if harness.is_running:
            harness.stop()


if __name__ == "__main__":
    example_usage()