#!/usr/bin/env python3
"""
Slow-Inference Agent Harness with External Polling Support
Provides a polling mechanism for external agents to check if slow-inference 
is still alive/running during extended wait times.
"""

import subprocess
import time
import os
import sys
import signal
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
import psutil


@dataclass
class HealthStatus:
    """Health status of a slow-inference process."""
    timestamp: float
    alive: bool
    pid: Optional[int]
    exit_code: Optional[int]
    uptime: Optional[float]
    memory_mb: Optional[float]
    cpu_percent: Optional[float]
    model_path: str
    prompt: str
    tokens_generated: Optional[int] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class SlowInferenceTask:
    """Represents a single slow-inference task."""
    
    def __init__(self, task_id: str, model_path: str, prompt: str, 
                 max_tokens: int, engine_path: str, json_output: bool = False):
        self.task_id = task_id
        self.model_path = model_path
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.engine_path = engine_path
        self.json_output = json_output
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.start_time: Optional[float] = None
        self.completion_time: Optional[float] = None
        self.result: Optional[Dict[str, Any]] = None
        self.health_history: List[HealthStatus] = []
        
    def start(self) -> subprocess.Popen:
        """Start the slow-inference process."""
        cmd = [
            self.engine_path,
            "--model", self.model_path,
        ]
        
        if self.prompt:
            cmd.extend(["--prompt", self.prompt])
        cmd.extend(["--tokens", str(self.max_tokens)])
        if self.json_output:
            cmd.append("--json")
            
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd="/home/reeds/slow-inference/engine"
        )
        
        self.is_running = True
        self.start_time = time.time()
        return self.process
    
    def health_check(self) -> HealthStatus:
        """Perform a health check and return status."""
        if not self.process:
            return HealthStatus(
                timestamp=time.time(),
                alive=False,
                pid=None,
                exit_code=None,
                uptime=None,
                memory_mb=None,
                cpu_percent=None,
                model_path=self.model_path,
                prompt=self.prompt,
                error="No process started"
            )
            
        poll_result = self.process.poll()
        if poll_result is None:
            # Process is running
            health = HealthStatus(
                timestamp=time.time(),
                alive=True,
                pid=self.process.pid,
                exit_code=None,
                uptime=time.time() - self.start_time if self.start_time else None,
                memory_mb=None,
                cpu_percent=None,
                model_path=self.model_path,
                prompt=self.prompt
            )
            
            try:
                ps_process = psutil.Process(self.process.pid)
                health.memory_mb = round(ps_process.memory_info().rss / 1024 / 1024, 2)
                health.cpu_percent = ps_process.cpu_percent(interval=0.05)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
            self.health_history.append(health)
            return health
        else:
            # Process has exited
            health = HealthStatus(
                timestamp=time.time(),
                alive=False,
                pid=self.process.pid,
                exit_code=poll_result,
                uptime=time.time() - self.start_time if self.start_time else None,
                memory_mb=None,
                cpu_percent=None,
                model_path=self.model_path,
                prompt=self.prompt
            )
            self.health_history.append(health)
            self.is_running = False
            self.completion_time = time.time()
            return health
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for completion with optional timeout."""
        if not self.process:
            return {"error": "No process started"}
            
        try:
            stdout, stderr = self.process.communicate(timeout=timeout)
            self.is_running = False
            self.completion_time = time.time()
            self.result = {
                "stdout": stdout,
                "stderr": stderr,
                "returncode": self.process.returncode,
                "timed_out": False
            }
            return self.result
        except subprocess.TimeoutExpired:
            self.stop()
            self.result = {
                "error": f"Process timed out after {timeout} seconds",
                "timed_out": True,
                "pid": self.process.pid if self.process else None
            }
            return self.result
    
    def stop(self):
        """Stop the process."""
        if not self.process:
            return
            
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            
        self.is_running = False
        self.process = None


class SlowInferenceHarness:
    """
    Main harness for managing slow-inference tasks with external polling support.
    """
    
    def __init__(self, 
                 model_path: str, 
                 engine_path: Optional[str] = None,
                 status_file: str = "/tmp/slow_inference_status.json"):
        self.model_path = model_path
        self.engine_path = engine_path or "/home/reeds/slow-inference/engine/target/release/slow-inference"
        self.status_file = Path(status_file)
        self.tasks: Dict[str, SlowInferenceTask] = {}
        self.lock = threading.Lock()
        
        # Validate paths
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"Engine not found at {self.engine_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        print(f"✓ Harness initialized")
        print(f"  Engine: {self.engine_path}")
        print(f"  Model: {self.model_path}")
        print(f"  Status file: {self.status_file}")
        
    def create_task(self, task_id: str, prompt: str = "", 
                    max_tokens: int = 50, json_output: bool = False) -> SlowInferenceTask:
        """Create and register a new task."""
        with self.lock:
            if task_id in self.tasks:
                raise ValueError(f"Task {task_id} already exists")
                
            task = SlowInferenceTask(
                task_id=task_id,
                model_path=self.model_path,
                prompt=prompt,
                max_tokens=max_tokens,
                engine_path=self.engine_path,
                json_output=json_output
            )
            self.tasks[task_id] = task
            return task
    
    def start_task(self, task_id: str) -> subprocess.Popen:
        """Start a registered task."""
        with self.lock:
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} not found")
            task = self.tasks[task_id]
            return task.start()
    
    def get_task(self, task_id: str) -> Optional[SlowInferenceTask]:
        """Get a task by ID."""
        with self.lock:
            return self.tasks.get(task_id)
    
    def list_tasks(self) -> List[Dict[str, Any]]:
        """List all tasks with their current status."""
        with self.lock:
            result = []
            for task_id, task in self.tasks.items():
                health = task.health_check()
                result.append({
                    "task_id": task_id,
                    "alive": health.alive,
                    "pid": health.pid,
                    "exit_code": health.exit_code,
                    "uptime": health.uptime,
                    "prompt": task.prompt[:50] + "..." if len(task.prompt) > 50 else task.prompt,
                    "max_tokens": task.max_tokens
                })
            return result
    
    def get_status(self, task_id: str) -> Optional[HealthStatus]:
        """Get current health status of a task."""
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            return task.health_check()
    
    def write_status_file(self):
        """Write current status to file for external polling."""
        with self.lock:
            status_data = {
                "timestamp": time.time(),
                "tasks": {}
            }
            for task_id, task in self.tasks.items():
                health = task.health_check()
                status_data["tasks"][task_id] = health.to_dict()
            
            # Write atomically
            temp_file = self.status_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(status_data, f, indent=2)
            temp_file.replace(self.status_file)
    
    def monitor_all(self, check_interval: float = 5.0, 
                    duration: Optional[float] = None):
        """Monitor all running tasks."""
        start_time = time.time()
        print(f"📊 Monitoring all tasks (check every {check_interval}s)")
        
        try:
            while True:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break
                    
                # Check all tasks
                with self.lock:
                    running_tasks = [t for t in self.tasks.values() if t.is_running]
                    
                if not running_tasks:
                    print("✅ All tasks completed")
                    break
                    
                for task in running_tasks:
                    health = task.health_check()
                    status = "🟢" if health.alive else "🔴"
                    info = f"{status} Task {task.task_id}: PID={health.pid}"
                    if health.memory_mb:
                        info += f" RAM={health.memory_mb}MB"
                    if health.cpu_percent:
                        info += f" CPU={health.cpu_percent:.1f}%"
                    if health.uptime:
                        info += f" UP={health.uptime:.0f}s"
                    print(info)
                
                # Write status file for external polling
                self.write_status_file()
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n👋 Monitoring interrupted")
        finally:
            self.write_status_file()
    
    def cleanup_completed(self):
        """Remove completed tasks from memory."""
        with self.lock:
            completed = [tid for tid, t in self.tasks.items() if not t.is_running]
            for tid in completed:
                del self.tasks[tid]
            if completed:
                print(f"🧹 Cleaned up {len(completed)} completed tasks")


def run_with_polling(model_path: str, prompt: str, max_tokens: int = 50,
                     check_interval: float = 5.0):
    """
    Convenience function: run a single task with health polling.
    Returns the result when complete.
    """
    harness = SlowInferenceHarness(model_path)
    task_id = f"task_{int(time.time())}"
    
    task = harness.create_task(task_id, prompt, max_tokens)
    harness.start_task(task_id)
    
    print(f"🚀 Started task {task_id} (PID: {task.process.pid})")
    
    try:
        while True:
            health = harness.get_status(task_id)
            if not health:
                print("❌ Task not found")
                return None
                
            if not health.alive:
                print(f"✅ Task completed (exit code: {health.exit_code})")
                # Get final result
                result = task.wait_for_completion(timeout=1)
                return result
            
            status = f"🟢 Alive | PID:{health.pid}"
            if health.memory_mb:
                status += f" RAM:{health.memory_mb}MB"
            if health.cpu_percent:
                status += f" CPU:{health.cpu_percent:.1f}%"
            if health.uptime:
                status += f" UP:{health.uptime:.0f}s"
            print(status)
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted, stopping task...")
        task.stop()
        return {"interrupted": True}
    finally:
        harness.write_status_file()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Slow-Inference Agent Harness")
    parser.add_argument("--model", default="/home/reeds/slow-inference/models/Tiny-Moe.Q4_K_M.gguf",
                        help="Path to GGUF model")
    parser.add_argument("--prompt", default="The future of AI is",
                        help="Prompt for generation")
    parser.add_argument("--tokens", type=int, default=10,
                        help="Max tokens to generate")
    parser.add_argument("--monitor", action="store_true",
                        help="Monitor all tasks continuously")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Polling interval in seconds")
    
    args = parser.parse_args()
    
    if args.monitor:
        # Monitor mode - run multiple tasks
        harness = SlowInferenceHarness(args.model)
        
        # Create multiple tasks
        for i in range(2):
            task_id = f"task_{i}"
            harness.create_task(task_id, args.prompt, args.tokens)
            harness.start_task(task_id)
            
        harness.monitor_all(check_interval=args.interval)
    else:
        # Single task mode
        result = run_with_polling(args.model, args.prompt, args.tokens, args.interval)
        if result:
            print(f"\n📋 Result: {json.dumps(result, indent=2)[:500]}...")