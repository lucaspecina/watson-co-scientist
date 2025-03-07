"""
Configuration management for the Co-Scientist system.
Handles loading and validating configuration from files and environment variables.
"""

import os
import json
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

class ModelConfig(BaseModel):
    """Configuration for a language model."""
    provider: str = Field(..., description="The provider of the language model (azure, openai, ollama, etc.)")
    model_name: str = Field(..., description="The name of the model to use")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    api_base: Optional[str] = Field(None, description="Base URL for the API")
    api_version: Optional[str] = Field(None, description="API version")
    deployment_id: Optional[str] = Field(None, description="Deployment ID (for Azure)")
    temperature: float = Field(0.7, description="Temperature for generation")
    max_tokens: int = Field(4000, description="Maximum number of tokens to generate")

class AgentConfig(BaseModel):
    """Configuration for an agent."""
    model: str = Field(..., description="The model to use for this agent")
    system_prompt: str = Field(..., description="System prompt for the agent")
    temperature: Optional[float] = Field(None, description="Temperature override for this agent")
    max_tokens: Optional[int] = Field(None, description="Max tokens override for this agent")

class SystemConfig(BaseModel):
    """Overall system configuration."""
    models: Dict[str, ModelConfig] = Field(..., description="Available models configuration")
    default_model: str = Field(..., description="Default model to use")
    agents: Dict[str, AgentConfig] = Field(..., description="Agent configurations")
    web_search_enabled: bool = Field(True, description="Whether web search is enabled")
    web_search_provider: str = Field("tavily", description="Web search provider to use (tavily, bing, serper)")
    web_search_api_key: Optional[str] = Field(None, description="API key for web search")
    tournament_iterations: int = Field(5, description="Number of iterations for tournaments")
    max_hypotheses: int = Field(100, description="Maximum number of hypotheses to generate")
    literature_search_depth: int = Field(5, description="Number of literature results to fetch per query")
    
    # Add dynamic attributes for runtime use
    model_config = {"extra": "allow"}
    
def load_default_config() -> Dict[str, Any]:
    """
    Load the default configuration with model settings from environment variables.
    
    Returns:
        Dict[str, Any]: Default configuration.
    """
    # Default config with environment variables
    default_config = {
        "models": {
            "azure-gpt4": {
                "provider": "azure",
                "model_name": "gpt-4",
                "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
                "api_base": os.environ.get("AZURE_OPENAI_ENDPOINT"),
                "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15"),
                "deployment_id": os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
                "temperature": 0.7,
                "max_tokens": 4000
            },
            "openai-gpt4": {
                "provider": "openai",
                "model_name": "gpt-4-turbo",
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "temperature": 0.7,
                "max_tokens": 4000
            },
            "ollama-llama3": {
                "provider": "ollama",
                "model_name": "llama3",
                "api_base": "http://localhost:11434",
                "temperature": 0.7,
                "max_tokens": 4000
            }
        },
        "default_model": "azure-gpt4",
        "agents": {
            "supervisor": {
                "model": "azure-gpt4",
                "system_prompt": "You are the Supervisor agent for the Co-Scientist system. Your role is to coordinate the work of specialized agents and manage system resources."
            },
            "generation": {
                "model": "azure-gpt4",
                "system_prompt": "You are the Generation agent for the Co-Scientist system. Your role is to generate novel research hypotheses and proposals based on the research goal."
            },
            "reflection": {
                "model": "azure-gpt4",
                "system_prompt": "You are the Reflection agent for the Co-Scientist system. Your role is to critically review research hypotheses and proposals for correctness, quality, novelty, and ethics."
            },
            "ranking": {
                "model": "azure-gpt4",
                "system_prompt": "You are the Ranking agent for the Co-Scientist system. Your role is to conduct tournaments to rank research hypotheses and proposals."
            },
            "proximity": {
                "model": "azure-gpt4",
                "system_prompt": "You are the Proximity agent for the Co-Scientist system. Your role is to calculate similarity between research hypotheses and proposals."
            },
            "evolution": {
                "model": "azure-gpt4",
                "system_prompt": "You are the Evolution agent for the Co-Scientist system. Your role is to improve existing research hypotheses and proposals."
            },
            "meta_review": {
                "model": "azure-gpt4",
                "system_prompt": "You are the Meta-review agent for the Co-Scientist system. Your role is to synthesize insights from reviews and tournaments to improve other agents' performance."
            }
        },
        "web_search_enabled": True,
        "web_search_provider": "tavily",
        "web_search_api_key": os.environ.get("TAVILY_API_KEY") or os.environ.get("BING_SEARCH_API_KEY"),
        "tournament_iterations": 5,
        "max_hypotheses": 100,
        "literature_search_depth": 5
    }
    
    return default_config

def load_config(config_name: str = "default") -> SystemConfig:
    """
    Load configuration from a JSON file or use the default if not found.
    
    Args:
        config_name (str): Name of the configuration to load.
        
    Returns:
        SystemConfig: The loaded configuration.
    """
    # Check if config file exists
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config")
    config_file = os.path.join(config_dir, f"{config_name}.json")
    
    # Load from file if it exists
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config_data = json.load(f)
        
        # Important: Override with environment variables
        if "models" in config_data and "azure-gpt4" in config_data["models"]:
            azure_config = config_data["models"]["azure-gpt4"]
            azure_config["api_key"] = os.environ.get("AZURE_OPENAI_API_KEY")
            azure_config["api_base"] = os.environ.get("AZURE_OPENAI_ENDPOINT")
            azure_config["api_version"] = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")
            azure_config["deployment_id"] = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    else:
        # Use default config
        config_data = load_default_config()
    
    # Create directory for config files if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)
    
    # Save default config if it doesn't exist, but NEVER save API keys to file
    if config_name == "default" and not os.path.exists(config_file):
        # Create a copy of the config data without API keys
        safe_config = json.loads(json.dumps(config_data))
        for model_name, model_config in safe_config["models"].items():
            if "api_key" in model_config:
                model_config["api_key"] = None
        safe_config["web_search_api_key"] = None
        
        with open(config_file, "w") as f:
            json.dump(safe_config, f, indent=2)
    
    # Create and validate the config
    return SystemConfig(**config_data)