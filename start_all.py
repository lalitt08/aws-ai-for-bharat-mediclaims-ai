#!/usr/bin/env python3
"""
MediClaims AI - Unified System Startup
=======================================

Starts BOTH systems as one prototype:
  1. Pre-Submission Pipeline  (MCP Server + Insurer APIs + Web Dashboard)
  2. Post-Submission Dashboard (Appeals Management on port 8003)
"""

import os
import sys
import time
import subprocess
import signal
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COMPONENTS = [
    {
        "name": "MCP Server",
        "cmd": [sys.executable, os.path.join(BASE_DIR, "mcp_server", "main.py")],
        "url": "http://localhost:8001",
        "wait": 4,
    },
    {
        "name": "Primary Insurance API (BlueCross/Aetna)",
        "cmd": [sys.executable, os.path.join(BASE_DIR, "tools", "insurer_api_primary.py")],
        "url": "http://localhost:8081",
        "wait": 3,
    },
    {
        "name": "Secondary Insurance API (Cigna/United)",
        "cmd": [sys.executable, os.path.join(BASE_DIR, "tools", "insurer_api_secondary.py")],
        "url": "http://localhost:8082",
        "wait": 3,
    },
    {
        "name": "Pre-Submission Dashboard",
        "cmd": [sys.executable, os.path.join(BASE_DIR, "web_dashboard", "api_server.py")],
        "url": "http://localhost:5000",
        "wait": 4,
    },
    {
        "name": "Post-Submission Appeals Dashboard",
        "cmd": [sys.executable, os.path.join(BASE_DIR, "post_submission_demo", "app.py")],
        "url": "http://localhost:8003",
        "wait": 3,
    },
]


def print_banner():
    print("=" * 70)
    print("  MediClaims AI - Unified Healthcare Claims Prototype")
    print("=" * 70)
    print()
    print("  Pre-Submission  : AI agents process & submit claims")
    print("  Post-Submission : Appeals dashboard for denied claims")
    print()
    print("=" * 70)


def check_env():
    required = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
                 "AZURE_OPENAI_DEPLOYMENT_NAME", "AZURE_OPENAI_API_VERSION"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(f"[WARN] Missing env vars: {missing}")
        print("       Some AI features may not work. Check your .env file.")
    else:
        print("[OK] All Azure OpenAI env vars set.")


def start_component(comp):
    try:
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
        proc = subprocess.Popen(comp["cmd"], **kwargs)
        time.sleep(comp["wait"])
        if proc.poll() is None:
            print(f"  [RUNNING] {comp['name']}  ->  {comp['url']}")
            return proc
        else:
            print(f"  [FAILED]  {comp['name']}")
            return None
    except Exception as e:
        print(f"  [ERROR]   {comp['name']}: {e}")
        return None


def main():
    print_banner()
    check_env()

    processes = []
    print("\nStarting components...\n")

    for comp in COMPONENTS:
        proc = start_component(comp)
        if proc:
            processes.append((comp["name"], proc))

    print("\n" + "=" * 70)
    if processes:
        print("  SYSTEM READY")
        print()
        print("  Pre-Submission Dashboard : http://localhost:5000")
        print("  Post-Submission Appeals  : http://localhost:8003")
        print("  MCP Server               : http://localhost:8001")
        print("  API Docs (Post-Sub)      : http://localhost:8003/docs")
        print()
        print("  WORKFLOW:")
        print("    1. Open http://localhost:5000 -> Submit a claim")
        print("    2. If denied, it appears at http://localhost:8003")
        print("    3. Review denial, generate appeal, resubmit")
        print()
        print("  Press Ctrl+C to stop everything.")
        print("=" * 70)
    else:
        print("  No components started. Check errors above.")
        return

    try:
        while True:
            time.sleep(2)
            dead = [n for n, p in processes if p.poll() is not None]
            if dead:
                print(f"\n[WARN] Stopped: {', '.join(dead)}")
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        for name, proc in processes:
            if proc.poll() is None:
                print(f"  Stopping {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("\nAll components stopped.")


if __name__ == "__main__":
    main()
