#!/usr/bin/env python3
"""
Healthcare Claims AI - Agentic System Startup
=============================================

This script starts the complete agentic system with MCP integration
and OpenAI-powered multi-agent processing.
"""

import os
import sys
import time
import subprocess
import signal
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def print_banner():
    """Print agentic system banner"""
    print("=" * 80)
    print("HEALTHCARE CLAIMS AI - AGENTIC SYSTEM")
    print("=" * 80)
    print("Multi-Agent AI Processing with MCP & OpenAI Integration")
    print("=" * 80)

def set_environment():
    """Set environment for agentic system"""
    os.environ['OPERATIONAL_MODE'] = 'mcp'
    print("[SUCCESS] Environment configured for agentic system")

def check_requirements():
    """Check if required environment variables are set"""
    print("[INFO] Checking environment variables...")
    
    required_vars = [
        'AZURE_OPENAI_API_KEY',
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_DEPLOYMENT_NAME',
        'AZURE_OPENAI_API_VERSION'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            print(f"[SUCCESS] {var} = {value[:20]}..." if len(value) > 20 else f"[SUCCESS] {var} = {value}")
    
    if missing_vars:
        print("[WARNING] Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file before starting.")
        print("The system will continue but some features may not work.")
        time.sleep(3)
        return True  # Continue anyway
    
    print("[SUCCESS] All required environment variables are set")
    return True

def start_component(name, command, wait_time=3):
    """Start a system component"""
    print(f"\n[STARTING] {name}...")
    
    try:
        process = subprocess.Popen(
            command,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
        )
        print(f"[WAIT] Waiting {wait_time} seconds for {name} to initialize...")
        time.sleep(wait_time)
        
        # Check if process is still running
        if process.poll() is None:
            print(f"[SUCCESS] {name} started successfully")
            return process
        else:
            print(f" {name} ")
            # Print error output
            stdout, stderr = process.communicate()
            if stderr:
                print(f" {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f" starting {name}: {e}")
        return None

def main():
    """Main function to start the agentic system"""
    print_banner()
    
    # Check requirements
    check_requirements()
    
    # Set environment
    set_environment()
    
    processes = []
    
    try:
        # Get the base directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        print("\n" + "=" * 60)
        print("STARTING AGENTIC SYSTEM COMPONENTS")
        print("=" * 60)
        
        # Start MCP Server
        print("\n1. Starting MCP Server...")
        mcp_process = start_component(
            "MCP Server", 
            [sys.executable, os.path.join(base_dir, 'mcp_server', 'main.py')],
            wait_time=5
        )
        if mcp_process:
            processes.append(("MCP Server", mcp_process))
            print("[INFO] MCP Server: http://localhost:8001")
        
        # Start Primary Insurance API
        print("\n2. Starting Primary Insurance API...")
        primary_api_process = start_component(
            "Primary Insurance API", 
            [sys.executable, os.path.join(base_dir, 'tools', 'insurer_api_primary.py')],
            wait_time=3
        )
        if primary_api_process:
            processes.append(("Primary Insurance API", primary_api_process))
            print("[INFO] Primary Insurance API: http://localhost:8081")
        
        # Start Secondary Insurance API
        print("\n3. Starting Secondary Insurance API...")
        secondary_api_process = start_component(
            "Secondary Insurance API", 
            [sys.executable, os.path.join(base_dir, 'tools', 'insurer_api_secondary.py')],
            wait_time=3
        )
        if secondary_api_process:
            processes.append(("Secondary Insurance API", secondary_api_process))
            print("[INFO] Secondary Insurance API: http://localhost:8082")
        
        # Start Web Dashboard (includes orchestrator functionality)
        print("\n4. Starting Web Dashboard...")
        dashboard_process = start_component(
            "Web Dashboard",
            [sys.executable, os.path.join(base_dir, 'web_dashboard', 'api_server.py')],
            wait_time=3
        )
        if dashboard_process:
            processes.append(("Web Dashboard", dashboard_process))
            print("[INFO] Web Dashboard: http://localhost:5000")
        
        if processes:
            print("\n" + "=" * 60)
            print("[SUCCESS] AGENTIC SYSTEM READY")
            print("=" * 60)
            
            print("\nSystem Components Running:")
            for name, process in processes:
                status = "[RUNNING]" if process.poll() is None else "[STOPPED]"
                print(f"  {name}: {status}")
            
            print("\nAccess Points:")
            print("  [WEB] Web Dashboard: http://localhost:5000")
            print("  [MCP] MCP Server: http://localhost:8001")
            print("  [API1] Primary Insurance API: http://localhost:8081")
            print("  [API2] Secondary Insurance API: http://localhost:8082")
            print("  [HEALTH] Health Check: http://localhost:8001/health")
            
            print("\n[STOP] Press Ctrl+C to stop the entire system")
            print("\n" + "=" * 60)
            
            # Keep the main process running
            while True:
                time.sleep(1)
                
                # Check if any process died
                dead_processes = []
                for name, process in processes:
                    if process.poll() is not None:
                        dead_processes.append(name)
                
                if dead_processes:
                    print(f"\n[WARNING] Component(s) stopped: {', '.join(dead_processes)}")
                    break
        else:
            print("\n No components started successfully")
            
    except KeyboardInterrupt:
        print("\n\n[STOP] Stopping agentic system...")
        
    except Exception as e:
        print(f"\n[ERROR] System error: {e}")
        
    finally:
        # Clean up all processes
        print("\n[CLEANUP] Cleaning up processes...")
        for name, process in processes:
            if process.poll() is None:
                print(f"  Stopping {name}...")
                process.terminate()
                time.sleep(1)
                if process.poll() is None:
                    process.kill()
        
        print("\n" + "=" * 80)
        print("[STOPPED] Healthcare Claims AI - Agentic System Stopped")
        print("=" * 80)

if __name__ == "__main__":
    main()
