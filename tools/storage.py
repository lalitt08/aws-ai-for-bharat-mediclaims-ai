# tools/storage.py

import os
import json
from datetime import datetime
import asyncio

STORAGE_DIR = "data/training"
LOG_FILE = os.path.join(STORAGE_DIR, "feedback_log.jsonl")

# Ensure storage directory exists
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

async def append_feedback_log(entry: dict) -> None:
    """
    Appends a claim processing summary to the feedback log file (JSONL format).
    """
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[Storage] Failed to write feedback log: {str(e)}")

async def read_feedback_logs(limit: int = 100) -> list:
    """
    Reads the most recent claim logs for analysis/retraining.
    """
    try:
        if not os.path.exists(LOG_FILE):
            return []

        async with asyncio.Lock():  # Optional: for concurrent file reads
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()[-limit:]
                return [json.loads(line) for line in lines]
    except Exception as e:
        print(f"[Storage] Failed to read feedback logs: {str(e)}")
        return []
