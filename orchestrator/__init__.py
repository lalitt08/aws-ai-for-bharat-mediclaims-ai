"""
Orchestrator package for agentic claims processing
Provides MCP client integration and agent coordination
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from orchestrator.mcp_client import MCPClient, mcp_client
    from orchestrator.orchestrator import AgentOrchestrator, orchestrator
    
    __all__ = ["MCPClient", "AgentOrchestrator", "mcp_client", "orchestrator"]
    
except ImportError as e:
    print(f"Warning: Could not import orchestrator components: {e}")
    MCPClient = None
    AgentOrchestrator = None
    mcp_client = None
    orchestrator = None
    
    __all__ = []
