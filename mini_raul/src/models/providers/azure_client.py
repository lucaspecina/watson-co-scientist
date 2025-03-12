"""
Azure OpenAI client implementation.
"""
import os
import logging
import json
from typing import Dict, Any, List, Optional, Union

from ..llm import LLMClient

logger = logging.getLogger(__name__)

try:
    from openai import AsyncAzureOpenAI
except ImportError:
    logger.warning("Azure OpenAI client not available. Please install openai package.")
    AsyncAzureOpenAI = None


class AzureOpenAIClient(LLMClient):
    """Client for the Azure OpenAI API."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Azure OpenAI client.
        
        Args:
            config: Optional configuration for the client.
        """
        super().__init__(config)
        
        # Load from environment variables if not provided
        api_key = self.config.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY")
        api_version = self.config.get("api_version") or os.getenv("AZURE_OPENAI_API_VERSION") or "2023-05-15"
        azure_endpoint = self.config.get("azure_endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = self.config.get("deployment_name") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        if not all([api_key, azure_endpoint, self.deployment_name]):
            logger.warning("Missing required Azure OpenAI configuration. Functionality will be limited.")
        
        # Initialize the client if dependencies are available
        if AsyncAzureOpenAI:
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint
            )
        else:
            self.client = None
    
    async def generate(self, 
                      prompt: str, 
                      system_message: Optional[str] = None,
                      temperature: float = 0.7,
                      max_tokens: Optional[int] = None) -> str:
        """
        Generate text using Azure OpenAI.
        
        Args:
            prompt: The prompt to send to the model.
            system_message: Optional system message to set context.
            temperature: Controls randomness. Higher means more random.
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            The generated text.
            
        Raises:
            RuntimeError: If the client is not initialized or generation fails.
        """
        if not self.client:
            raise RuntimeError("Azure OpenAI client not initialized. Please check configuration.")
        
        try:
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            messages.append({"role": "user", "content": prompt})
            
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating text with Azure OpenAI: {e}")
            raise RuntimeError(f"Failed to generate text: {e}") 