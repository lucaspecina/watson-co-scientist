# Watson Co-Scientist

An AI system that assists scientists in generating, refining, and testing research hypotheses. This system serves as a collaborative partner, enhancing scientists' ability to explore complex scientific problems efficiently.

## Overview

The co-scientist is designed to act as a helpful assistant and collaborator to scientists, helping to accelerate the scientific discovery process. The system is a compound, multi-agent AI system built on large language models, designed to mirror the reasoning process underpinning the scientific method.

Given a research goal specified in natural language, the system can:
- Generate novel, original research hypotheses
- Propose experimental protocols for downstream validations
- Provide rigorous verification of hypotheses through causal reasoning and computational simulation
- Identify specific, testable predictions and potential failure modes
- Provide grounding for recommendations by citing relevant literature
- Explain the reasoning behind proposals with probabilistic assessment

## System Architecture

The system employs a multi-agent architecture integrated within an asynchronous task execution framework:

1. **Supervisor Agent**: Coordinates the work of specialized agents and manages system resources
2. **Generation Agent**: Generates novel research hypotheses and research proposals
3. **Reflection Agent**: Performs multiple types of reviews, including:
   - Standard reviews for correctness, quality, novelty, and ethics
   - Deep verification reviews with causal reasoning and probabilistic assessment
   - Simulation reviews with computational modeling and prediction testing
   - Observation reviews linking hypotheses to existing experimental evidence
   - Protocol reviews assessing the feasibility and rigor of experimental designs
4. **Ranking Agent**: Conducts tournaments to rank hypotheses via scientific debates
5. **Proximity Agent**: Calculates similarity between hypotheses to organize the hypothesis space
6. **Evolution Agent**: Improves existing hypotheses through various strategies
7. **Meta-Review Agent**: Synthesizes insights from reviews and tournaments into a comprehensive research overview, including causal reasoning patterns and verification recommendations

## Features

- **Scientist-in-the-loop**: The system is designed for collaboration with scientists, allowing them to guide the process
- **Multi-agent architecture**: Specialized agents work together to generate, evaluate, and refine hypotheses
- **Iterative improvement**: The system continuously refines hypotheses based on feedback and evaluation
- **Deep verification capabilities**: Rigorous assessment of hypotheses through causal reasoning, assumption analysis, and probabilistic evaluation
- **Computational simulation**: Modeling of hypotheses to test predictions, analyze sensitivity to parameters, and identify emergent properties
- **Domain-specific knowledge integration**: Grounds hypotheses in specialized knowledge from scientific domains
- **Cross-domain inspiration**: Applies concepts from other scientific domains to inspire innovative thinking
- **Adaptive evolution strategies**: Intelligently selects improvement approaches based on review patterns
- **Scientific ontology integration**: Utilizes domain ontologies to connect concepts and relationships
- **Experimental protocol generation**: Development of detailed, practical protocols to test hypotheses in the lab
- **Tournament-based ranking**: Hypotheses are evaluated and ranked through simulated scientific debates
- **Enhanced research overviews**: Comprehensive summaries with causal structures, verification approaches, and testable predictions
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
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name

# OpenAI (fallback)
OPENAI_API_KEY=your_openai_api_key

# Web Search (optional)
BING_SEARCH_API_KEY=your_bing_search_api_key
```

Note: The environment variable names have been updated to match Azure OpenAI's standard naming conventions.

### Configuration

The system uses a configuration file to manage settings for models and agents. A template is provided in `config/template.json`. If you want to customize the configuration:

1. Copy `config/template.json` to `config/default.json` (this file is git-ignored)
2. Edit the settings as needed

The system will automatically load configuration from environment variables, so you typically don't need to modify the config file unless you want to change model settings, agent prompts, or other parameters.

## Usage

### Command Line Interface (CLI)

The CLI provides an interactive interface to work with the system directly:

```bash
# Start the interactive CLI
python main.py

# Analyze a specific research goal
python main.py --research_goal "Investigate the molecular mechanisms of protein misfolding in neurodegenerative diseases."

# Use a specific configuration
python main.py --config custom_config
```

#### Interactive Mode Commands

When running in interactive mode, you can use the following commands:

- `goal: <text>` - Set a new research goal
- `run` - Run 1 iteration
- `run <N>` - Run N iterations
- `state` - Print the current system state
- `overview` - Generate and print a research overview
- `help` - Show help information
- `exit` or `quit` - Exit the program

Example session:
```
> goal: Investigate the role of mitochondrial dysfunction in neurodegenerative diseases
Analyzing research goal: Investigating the role of mitochondrial dysfunction in neurodegenerative...
Research goal set. Type 'run' to start processing, or 'run N' to run N iterations.

> run 2
Running 2 iteration(s)...
Completed iteration 1/2
Completed iteration 2/2

> state
============ CURRENT STATE ============
Research Goal: Investigate the role of mitochondrial dysfunction in neurodegenerative diseases...
Iterations completed: 2
Hypotheses generated: 6
Reviews completed: 4
Tournament matches: 3

Top Hypotheses:
  1. Mitochondrial Complex I Dysfunction in Parkinson's Disease (Rating: 1215.0)
  2. Mitochondrial DNA Mutations in Alzheimer's Disease (Rating: 1208.4)
  ...

> overview
Generating research overview...
============ RESEARCH OVERVIEW ============
Title: Mitochondrial Dysfunction in Neurodegenerative Diseases: Research Areas and Hypotheses
Summary: This overview synthesizes current research directions on the role of mitochondrial...
...
```

### API Server

The API server enables:
- Integration with other applications and workflows
- Remote access and control of the Co-Scientist system
- Parallel processing of multiple research goals
- Long-running background tasks without blocking the UI

#### Starting the API Server

```bash
# Start the API server with default settings (localhost:8000)
python api_server.py

# Specify host and port
python api_server.py --host 0.0.0.0 --port 8888
```

#### API Documentation

Once running, you can:
- Access the API documentation: `http://localhost:8000/docs`
- Browse the interactive Swagger UI to test endpoints

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/research_goal` | POST | Submit a new research goal |
| `/state` | GET | Get the current system state |
| `/run` | POST | Run system iterations in the background |
| `/hypotheses` | GET | Get generated hypotheses (with pagination) |
| `/overview` | GET | Get the latest research overview |

#### Example API Usage

Using curl:
```bash
# Submit a research goal
curl -X POST "http://localhost:8000/research_goal" \
  -H "Content-Type: application/json" \
  -d '{"text": "Investigate the role of mitochondrial dysfunction in neurodegenerative diseases"}'

# Run iterations
curl -X POST "http://localhost:8000/run" \
  -H "Content-Type: application/json" \
  -d '{"iterations": 3}'

# Get the current state
curl -X GET "http://localhost:8000/state"

# Get the top hypotheses (with pagination)
curl -X GET "http://localhost:8000/hypotheses?limit=5&offset=0"

# Get the latest research overview
curl -X GET "http://localhost:8000/overview"
```

Using Python requests:
```python
import requests
import json

# Submit a research goal
response = requests.post(
    "http://localhost:8000/research_goal",
    json={"text": "Investigate the role of mitochondrial dysfunction in neurodegenerative diseases"}
)
research_goal = response.json()
print(f"Research goal ID: {research_goal['id']}")

# Run iterations
requests.post(
    "http://localhost:8000/run",
    json={"iterations": 3}
)

# Get the latest state
state = requests.get("http://localhost:8000/state").json()
print(f"Iterations completed: {state['iterations_completed']}")
print(f"Hypotheses generated: {state['num_hypotheses']}")
```

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