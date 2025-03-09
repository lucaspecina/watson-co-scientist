"""
LLM provider interfaces for different AI model providers.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

from ..config.config import ModelConfig

logger = logging.getLogger("co_scientist")

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, 
                     messages: List[Dict[str, str]], 
                     temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            messages (List[Dict[str, str]]): List of messages in the conversation.
            temperature (Optional[float]): Temperature for generation.
            max_tokens (Optional[int]): Maximum number of tokens to generate.
            
        Returns:
            Dict[str, Any]: The model's response.
        """
        pass
        
    async def generate_text(self,
                         prompt: str,
                         system_prompt: str = "You are a helpful assistant.",
                         temperature: Optional[float] = None,
                         max_tokens: Optional[int] = None,
                         model: Optional[str] = None) -> str:
        """
        Generate text from a prompt using a simplified interface.
        
        Args:
            prompt (str): The prompt to send to the model.
            system_prompt (str, optional): The system prompt to use.
            temperature (Optional[float], optional): The temperature to use.
            max_tokens (Optional[int], optional): Maximum number of tokens to generate.
            model (Optional[str], optional): Override model name (not used for all providers).
            
        Returns:
            str: The generated text.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response["content"]
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating text: {str(e)}"
    
    @classmethod
    def create_provider(cls, config: ModelConfig) -> 'LLMProvider':
        """
        Factory method to create the appropriate provider instance.
        
        Args:
            config (ModelConfig): The model configuration.
            
        Returns:
            LLMProvider: An instance of the appropriate provider.
            
        Raises:
            ValueError: If the provider is not supported.
        """
        provider = config.provider.lower()
        
        if provider == "azure":
            return AzureOpenAIProvider(config)
        elif provider == "openai":
            return OpenAIProvider(config)
        elif provider == "ollama":
            return OllamaProvider(config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

class AzureOpenAIProvider(LLMProvider):
    """Provider for Azure OpenAI API."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the Azure OpenAI provider.
        
        Args:
            config (ModelConfig): The model configuration.
        """
        self.config = config
        
        # Validate required fields
        if not config.api_key:
            raise ValueError("Azure OpenAI API key is required")
        if not config.api_base:
            raise ValueError("Azure OpenAI API base URL is required")
        if not config.api_version:
            raise ValueError("Azure OpenAI API version is required")
        if not config.deployment_id:
            raise ValueError("Azure OpenAI deployment ID is required")
        
        # Import here to avoid requiring OpenAI for all providers
        from openai import AsyncAzureOpenAI
        
        # Initialize client
        self.client = AsyncAzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.api_base
        )
        
    async def generate(self, 
                     messages: List[Dict[str, str]], 
                     temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a response from Azure OpenAI.
        
        Args:
            messages (List[Dict[str, str]]): List of messages in the conversation.
            temperature (Optional[float]): Temperature for generation.
            max_tokens (Optional[int]): Maximum number of tokens to generate.
            
        Returns:
            Dict[str, Any]: The model's response.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.config.deployment_id,
                messages=messages,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            )
            
            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            logger.error(f"Error generating response from Azure OpenAI: {e}")
            raise

class OpenAIProvider(LLMProvider):
    """Provider for OpenAI API."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the OpenAI provider.
        
        Args:
            config (ModelConfig): The model configuration.
        """
        self.config = config
        
        # Validate required fields
        if not config.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Import here to avoid requiring OpenAI for all providers
        from openai import AsyncOpenAI
        
        # Initialize client
        self.client = AsyncOpenAI(
            api_key=config.api_key
        )
        
    async def generate(self, 
                     messages: List[Dict[str, str]], 
                     temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a response from OpenAI.
        
        Args:
            messages (List[Dict[str, str]]): List of messages in the conversation.
            temperature (Optional[float]): Temperature for generation.
            max_tokens (Optional[int]): Maximum number of tokens to generate.
            
        Returns:
            Dict[str, Any]: The model's response.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            )
            
            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
            raise

class OllamaProvider(LLMProvider):
    """Provider for Ollama API."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the Ollama provider.
        
        Args:
            config (ModelConfig): The model configuration.
        """
        self.config = config
        
        # Initialize httpx client
        import httpx
        self.api_base = config.api_base or "http://localhost:11434"
        self.client = httpx.AsyncClient(base_url=self.api_base)
        
    async def generate(self, 
                      messages: List[Dict[str, str]], 
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a response from Ollama.
        
        Args:
            messages (List[Dict[str, str]]): List of messages in the conversation.
            temperature (Optional[float]): Temperature for generation.
            max_tokens (Optional[int]): Maximum number of tokens to generate.
            
        Returns:
            Dict[str, Any]: The model's response.
        """
        try:
            # Convert the messages format to what Ollama expects
            formatted_messages = []
            system_msg = None
            for msg in messages:
                role = msg["role"]
                if role == "system":
                    # Add as system instruction
                    system_msg = msg["content"]
                else:
                    # Add as regular message
                    formatted_messages.append({
                        "role": role,
                        "content": msg["content"]
                    })
            
            # Prepare the request
            request_data = {
                "model": self.config.model_name,
                "messages": formatted_messages,
                "options": {
                    "temperature": temperature if temperature is not None else self.config.temperature,
                }
            }
            
            if system_msg:
                request_data["system"] = system_msg
                
            if max_tokens:
                request_data["options"]["num_predict"] = max_tokens
            
            # Make the request
            response = await self.client.post(
                "/api/chat",
                json=request_data
            )
            response.raise_for_status()
            result = response.json()
            
            return {
                "content": result["message"]["content"],
                "model": self.config.model_name,
                "usage": {
                    "prompt_tokens": 0,  # Ollama doesn't provide token usage
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {e}")
            raise