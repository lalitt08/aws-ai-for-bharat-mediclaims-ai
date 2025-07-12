"""
MCP (Model Context Protocol) Client for Healthcare Claims Processing
Handles communication with external tools and services
"""

import httpx
import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MCPClient:
    """MCP client for external tool integration"""
    
    def __init__(self, server_url: str = "http://localhost:8001"):
        self.server_url = server_url
        self.session_id = None
        self.client = httpx.AsyncClient()
        self.connected = False
    
    async def connect(self) -> bool:
        """Connect to MCP server"""
        try:
            response = await self.client.get(f"{self.server_url}/health")
            if response.status_code == 200:
                self.connected = True
                self.session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                logger.info(f"Connected to MCP server: {self.server_url}")
                return True
            else:
                logger.error(f"Failed to connect to MCP server: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"MCP connection error: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        if self.client:
            await self.client.aclose()
        self.connected = False
        logger.info("Disconnected from MCP server")
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a tool on the MCP server"""
        if not self.connected:
            await self.connect()
        
        try:
            request_data = {
                "tool_name": tool_name,
                "arguments": arguments,
                "session_id": self.session_id
            }
            
            response = await self.client.post(
                f"{self.server_url}/execute",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    logger.info(f"Tool executed successfully: {tool_name}")
                    return result.get("result")
                else:
                    logger.error(f"Tool execution failed: {result.get('error')}")
                    return None
            else:
                logger.error(f"HTTP error executing tool: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return None
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call a tool on the MCP server (alias for execute_tool)"""
        return await self.execute_tool(tool_name, arguments)
    
    async def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        if not self.connected:
            await self.connect()
        
        try:
            response = await self.client.get(f"{self.server_url}/tools")
            if response.status_code == 200:
                result = response.json()
                return result.get("tools", [])
            else:
                logger.error(f"Failed to get tools: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting tools: {e}")
            return []
    
    async def get_patient_data(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get patient data via MCP"""
        return await self.execute_tool("get_patient_data", {"patient_id": patient_id})
    
    async def validate_claim(self, claim_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate claim data via MCP"""
        return await self.execute_tool("validate_claim", {"claim_data": claim_data})
    
    async def submit_to_insurer(self, claim_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Submit claim to insurer via MCP"""
        return await self.execute_tool("submit_to_insurer", {"claim_data": claim_data})
    
    async def generate_appeal(self, denial_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate appeal via MCP"""
        return await self.execute_tool("generate_appeal", {"denial_data": denial_data})
    
    async def get_risk_score(self, claim_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get risk score via MCP"""
        return await self.execute_tool("get_risk_score", {"claim_data": claim_data})
    
    async def format_claim_data(self, raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Format claim data via MCP"""
        return await self.execute_tool("format_claim_data", {"raw_data": raw_data})
    
    async def update_denial_patterns(self, denial_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update denial patterns via MCP"""
        return await self.execute_tool("update_denial_patterns", {"denial_data": denial_data})

# Global MCP client instance
mcp_client = MCPClient()
