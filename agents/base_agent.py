"""
Base Agent Class for Healthcare Claims Processing
================================================

This module provides a base class for all agents in the system that
automatically adapts to the operational mode (standalone vs MCP).
"""

import asyncio
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from abc import ABC, abstractmethod

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.data_access import UnifiedDataAccess
from tools.logger import secure_log
from config.settings import Settings

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Base class for all agents in the healthcare claims system
    
    Provides unified data access and mode awareness for all agents
    """
    
    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        self.mode = self.settings.OPERATIONAL_MODE
        self.data_access = UnifiedDataAccess(self.settings)
        self.agent_name = self.__class__.__name__
        self.session_id = None
        
        secure_log(f"Initialized {self.agent_name} in {self.mode} mode", "INFO")
    
    async def initialize(self):
        """Initialize the agent (override in subclasses if needed)"""
        secure_log(f"{self.agent_name} initialized successfully", "INFO")
    
    @abstractmethod
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method - must be implemented by subclasses
        
        Args:
            data: Input data for processing
            
        Returns:
            Dict containing processing results
        """
        pass
    
    async def get_patient_data(self, patient_id: str) -> Dict[str, Any]:
        """Get patient data using unified access"""
        return await self.data_access.get_patient_data(patient_id)
    
    async def validate_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate claim data using unified access"""
        return await self.data_access.validate_claim(claim_data)
    
    async def submit_to_insurer(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit claim to insurer using unified access"""
        return await self.data_access.submit_to_insurer(claim_data)
    
    async def get_risk_score(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get risk score using unified access"""
        return await self.data_access.get_risk_score(patient_data)
    
    def get_mode_info(self) -> Dict[str, Any]:
        """Get information about current operational mode"""
        return {
            "agent_name": self.agent_name,
            "mode": self.mode,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def set_session_id(self, session_id: str):
        """Set session ID for tracking"""
        self.session_id = session_id
    
    async def log_activity(self, activity: str, level: str = "INFO"):
        """Log agent activity"""
        secure_log(f"[{self.agent_name}] {activity}", level)


class StandaloneAgent(BaseAgent):
    """
    Agent specifically designed for standalone mode operation
    Forces standalone mode regardless of settings
    """
    
    def __init__(self, settings: Settings = None):
        # Override settings to force standalone mode
        if settings is None:
            settings = Settings()
        
        # Create a copy of settings with forced standalone mode
        class StandaloneSettings:
            def __init__(self, base_settings):
                for attr in dir(base_settings):
                    if not attr.startswith('_'):
                        setattr(self, attr, getattr(base_settings, attr))
                # Force standalone mode
                self.OPERATIONAL_MODE = 'standalone'
        
        standalone_settings = StandaloneSettings(settings)
        super().__init__(standalone_settings)


class MCPAgent(BaseAgent):
    """
    Agent specifically designed for MCP mode operation
    Forces MCP mode regardless of settings
    """
    
    def __init__(self, settings: Settings = None):
        # Override settings to force MCP mode
        if settings is None:
            settings = Settings()
        
        # Create a copy of settings with forced MCP mode
        class MCPSettings:
            def __init__(self, base_settings):
                for attr in dir(base_settings):
                    if not attr.startswith('_'):
                        setattr(self, attr, getattr(base_settings, attr))
                # Force MCP mode
                self.OPERATIONAL_MODE = 'mcp'
        
        mcp_settings = MCPSettings(settings)
        super().__init__(mcp_settings)
        
    async def initialize(self):
        """Initialize MCP agent with connection check"""
        try:
            # Test MCP connection
            mode_info = self.data_access.get_mode_info()
            secure_log(f"MCP Agent initialized: {mode_info}", "INFO")
        except Exception as e:
            secure_log(f"MCP Agent initialization failed: {e}", "ERROR")
            raise


class AgentFactory:
    """Factory for creating agents based on operational mode"""
    
    @staticmethod
    def create_agent(agent_class, settings: Settings = None, force_mode: str = None):
        """
        Create an agent instance based on operational mode
        
        Args:
            agent_class: The agent class to instantiate
            settings: Settings object (optional)
            force_mode: Force specific mode ('standalone' or 'mcp')
            
        Returns:
            Agent instance
        """
        if settings is None:
            settings = Settings()
        
        # Force mode if specified
        if force_mode:
            class ForcedSettings:
                def __init__(self, base_settings, forced_mode):
                    for attr in dir(base_settings):
                        if not attr.startswith('_'):
                            setattr(self, attr, getattr(base_settings, attr))
                    self.OPERATIONAL_MODE = forced_mode
            
            settings = ForcedSettings(settings, force_mode)
        
        return agent_class(settings)


# Example usage and testing
if __name__ == "__main__":
    # Test the base agent functionality
    
    class TestAgent(BaseAgent):
        async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "status": "processed",
                "agent": self.agent_name,
                "mode": self.mode,
                "data": data
            }
    
    async def test_agents():
        # Test standalone agent
        standalone_agent = AgentFactory.create_agent(TestAgent, force_mode='standalone')
        await standalone_agent.initialize()
        
        result = await standalone_agent.process({"test": "data"})
        print(f"Standalone agent result: {result}")
        
        # Test MCP agent (will fail if MCP server not running)
        try:
            mcp_agent = AgentFactory.create_agent(TestAgent, force_mode='mcp')
            await mcp_agent.initialize()
            result = await mcp_agent.process({"test": "data"})
            print(f"MCP agent result: {result}")
        except Exception as e:
            print(f"MCP agent test failed (expected if MCP server not running): {e}")
    
    # Run test
    asyncio.run(test_agents())
