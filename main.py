#!/usr/bin/env python3
"""
MCP RAG Server
==============

Main entry point for the MCP RAG Server application.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cli import app

if __name__ == "__main__":
    app()

