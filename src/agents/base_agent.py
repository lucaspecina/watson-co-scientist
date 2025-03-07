"""
Base agent implementation for the Co-Scientist system.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

from ..config.config import SystemConfig, AgentConfig
from ..core.llm_provider import LLMProvider

logger = logging.getLogger("co_scientist")

class BaseAgent:
    """Base class for all agents in the Co-Scientist system."""
    
    def __init__(self, name: str, config: SystemConfig):
        """
        Initialize the base agent.
        
        Args:
            name (str): The name of the agent.
            config (SystemConfig): The system configuration.
        """
        self.name = name
        self.system_config = config
        
        # Check if the agent has a configuration
        if name not in config.agents:
            raise ValueError(f"Agent {name} not found in configuration")
            
        self.agent_config = config.agents[name]
        
        # Get the model configuration
        model_name = self.agent_config.model
        if model_name not in config.models:
            raise ValueError(f"Model {model_name} not found in configuration")
            
        self.model_config = config.models[model_name]
        
        # Create the LLM provider
        self.provider = LLMProvider.create_provider(self.model_config)
        
        logger.info(f"Initialized {self.name} agent with model {model_name}")
        
    async def generate(self, 
                   prompt: str, 
                   system_prompt: Optional[str] = None,
                   temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None) -> str:
        """
        Generate a response from the agent.
        
        Args:
            prompt (str): The prompt to send to the agent.
            system_prompt (Optional[str]): The system prompt to use. Defaults to the agent's configuration.
            temperature (Optional[float]): The temperature to use. Defaults to the agent's configuration.
            max_tokens (Optional[int]): The maximum number of tokens to generate. Defaults to the agent's configuration.
            
        Returns:
            str: The agent's response.
        """
        # Use the provided system prompt or the agent's configuration
        _system_prompt = system_prompt or self.agent_config.system_prompt
        
        # Use the provided temperature or the agent's configuration or the model's configuration
        _temperature = temperature or self.agent_config.temperature or self.model_config.temperature
        
        # Use the provided max_tokens or the agent's configuration or the model's configuration
        _max_tokens = max_tokens or self.agent_config.max_tokens or self.model_config.max_tokens
        
        # Create the messages
        messages = [
            {"role": "system", "content": _system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Log the request
        logger.debug(f"Generating response for {self.name} agent with model {self.model_config.model_name}")
        
        # Generate the response
        response = await self.provider.generate(
            messages=messages,
            temperature=_temperature,
            max_tokens=_max_tokens
        )
        
        # Log the response
        logger.debug(f"Generated response for {self.name} agent: {response['usage']}")
        
        return response["content"]