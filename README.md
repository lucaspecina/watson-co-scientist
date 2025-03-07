# Watson Co-Scientist

An AI system that assists scientists in generating, refining, and testing research hypotheses. This system serves as a collaborative partner, enhancing scientists' ability to explore complex scientific problems efficiently.

## Overview

The co-scientist is designed to act as a helpful assistant and collaborator to scientists, helping to accelerate the scientific discovery process. The system is a compound, multi-agent AI system built on large language models, designed to mirror the reasoning process underpinning the scientific method.

Given a research goal specified in natural language, the system can:
- Generate novel, original research hypotheses
- Propose experimental protocols for downstream validations
- Provide grounding for recommendations by citing relevant literature
- Explain the reasoning behind proposals

## System Architecture

The system employs a multi-agent architecture integrated within an asynchronous task execution framework:

1. **Supervisor Agent**: Coordinates the work of specialized agents and manages system resources
2. **Generation Agent**: Generates novel research hypotheses and research proposals
3. **Reflection Agent**: Reviews and evaluates hypotheses for correctness, quality, novelty, and ethics
4. **Ranking Agent**: Conducts tournaments to rank hypotheses via scientific debates
5. **Proximity Agent**: Calculates similarity between hypotheses to organize the hypothesis space
6. **Evolution Agent**: Improves existing hypotheses through various strategies
7. **Meta-Review Agent**: Synthesizes insights from reviews and tournaments into a comprehensive research overview

## Features

- **Scientist-in-the-loop**: The system is designed for collaboration with scientists, allowing them to guide the process
- **Multi-agent architecture**: Specialized agents work together to generate, evaluate, and refine hypotheses
- **Iterative improvement**: The system continuously refines hypotheses based on feedback and evaluation
- **Tournament-based ranking**: Hypotheses are evaluated and ranked through simulated scientific debates
- **Research overview generation**: The system synthesizes findings into a comprehensive research overview
- **Multiple model providers**: Support for Azure OpenAI, OpenAI, and Ollama as LLM providers

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/watson-co-scientist.git
cd watson-co-scientist

# Create and activate a conda environment
conda create -n co_scientist python=3.11
conda activate co_scientist

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# Azure OpenAI (default provider)
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_API_BASE=https://your-resource-name.openai.azure.com
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_DEPLOYMENT_ID=your_deployment_id

# OpenAI (fallback)
OPENAI_API_KEY=your_openai_api_key

# Web Search (optional)
BING_SEARCH_API_KEY=your_bing_search_api_key
```

### Configuration

The system uses a configuration file to manage settings for models and agents. A template is provided in `config/template.json`. If you want to customize the configuration:

1. Copy `config/template.json` to `config/default.json` (this file is git-ignored)
2. Edit the settings as needed

The system will automatically load configuration from environment variables, so you typically don't need to modify the config file unless you want to change model settings, agent prompts, or other parameters.

## Usage

### Command Line Interface

```bash
# Start the interactive CLI
python main.py

# Analyze a specific research goal
python main.py --research_goal "Investigate the molecular mechanisms of protein misfolding in neurodegenerative diseases."
```

### API Server

```bash
# Start the API server
python api_server.py

# Access the API documentation at http://localhost:8000/docs
```

### API Endpoints

- `POST /research_goal`: Submit a research goal
- `GET /state`: Get the current system state
- `POST /run`: Run system iterations
- `GET /hypotheses`: Get generated hypotheses
- `GET /overview`: Get the latest research overview

## Development

### Project Structure

```
watson-co-scientist/
├── main.py                 # Main entry point
├── api_server.py           # API server script
├── requirements.txt        # Dependencies
├── config/                 # Configuration files
├── data/                   # Data storage
├── logs/                   # Log files
├── src/
│   ├── agents/             # Specialized agents
│   ├── api/                # API endpoints
│   ├── core/               # Core system components
│   ├── config/             # Configuration handling
│   ├── tools/              # External tools (web search, etc.)
│   └── utils/              # Utility functions
└── tests/                  # Test cases
```

### Running Tests

```bash
# Run tests
pytest tests/
```

## License

MIT