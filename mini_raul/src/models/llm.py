"""
LLM client module for mini-RAUL.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import os
import logging

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Base class for LLM clients."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM client.
        
        Args:
            config: Optional configuration for the client.
        """
        self.config = config or {}
    
    @abstractmethod
    async def generate(self, 
                      prompt: str, 
                      system_message: Optional[str] = None,
                      temperature: float = 0.7,
                      max_tokens: Optional[int] = None) -> str:
        """
        Generate text from the LLM.
        
        Args:
            prompt: The prompt to send to the model.
            system_message: Optional system message to set context.
            temperature: Controls randomness. Higher means more random.
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            The generated text.
        """
        pass


class LLMProvider:
    """Factory for LLM clients based on configuration."""
    
    @staticmethod
    def get_client(provider: str = "azure", config: Optional[Dict[str, Any]] = None) -> LLMClient:
        """
        Get an LLM client for the specified provider.
        
        Args:
            provider: The provider to use ("azure", "openai", "ollama").
            config: Optional configuration for the client.
            
        Returns:
            An LLM client.
            
        Raises:
            ValueError: If the provider is not supported.
        """
        if provider == "azure":
            from .providers.azure_client import AzureOpenAIClient
            return AzureOpenAIClient(config)
        elif provider == "openai":
            from .providers.openai_client import OpenAIClient
            return OpenAIClient(config)
        elif provider == "ollama":
            from .providers.ollama_client import OllamaClient
            return OllamaClient(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}") 