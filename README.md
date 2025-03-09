# Watson Co-Scientist

An AI system that assists scientists in generating, refining, and testing research hypotheses. This system serves as a collaborative partner, enhancing scientists' ability to explore complex scientific problems efficiently through multi-database scientific knowledge integration.

## Overview

The co-scientist is designed to act as a helpful assistant and collaborator to scientists, helping to accelerate the scientific discovery process. The system is a compound, multi-agent AI system built on large language models, designed to mirror the reasoning process underpinning the scientific method.

Given a research goal specified in natural language, the system can:
- Generate novel, original research hypotheses using knowledge from multiple scientific databases
- Propose experimental protocols for downstream validations
- Provide rigorous verification of hypotheses through causal reasoning and computational simulation
- Identify specific, testable predictions and potential failure modes
- Provide grounding for recommendations by citing relevant literature from multiple sources
- Explain the reasoning behind proposals with probabilistic assessment
- Synthesize knowledge across scientific domains to promote interdisciplinary discovery

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

- **Deep Scientific Database Integration**: 
  - Direct integration with PubMed, ArXiv, PubChem, UniProt, and other scientific databases
  - Cross-domain knowledge synthesis across multiple sources
  - Domain-specific knowledge retrieval and citation formatting
  - Knowledge Synthesizer for consolidating information from diverse sources
  - Intelligent source prioritization and relevance scoring

- **Advanced Knowledge Processing**:
  - Multi-domain concurrent search and retrieval
  - Automatic domain detection for research questions
  - Cross-domain relationship mapping for interdisciplinary research
  - Dynamic knowledge weighting based on research context
  - Knowledge synthesis with key concept extraction and connection detection
  - Scientific entity and relationship discovery across multiple sources

- **Scientist-in-the-loop**: The system is designed for collaboration with scientists, allowing them to guide the process
- **Multi-agent architecture**: Specialized agents work together to generate, evaluate, and refine hypotheses
- **Session persistence**: Support for resuming research sessions across multiple days or sessions
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
TAVILY_API_KEY=your_tavily_api_key
```

Note: The environment variable names have been updated to match Azure OpenAI's standard naming conventions.

### Optional Dependencies

The Watson Co-Scientist system includes a powerful Paper Knowledge Extraction System that benefits from several optional dependencies:

1. **PyMuPDF (fitz)**: Enhances PDF processing capabilities for more accurate text and structure extraction
   ```bash
   pip install PyMuPDF
   ```

2. **Tesseract OCR**: Enables text extraction from images and figures in papers
   ```bash
   pip install pytesseract
   # Also requires Tesseract to be installed on the system
   # macOS: brew install tesseract
   # Ubuntu: apt-get install tesseract-ocr
   ```

3. **OpenCV**: Provides advanced image processing for figure extraction and analysis
   ```bash
   pip install opencv-python
   ```

The system implements graceful degradation if these dependencies are not available, but installing them will significantly enhance the paper extraction capabilities.

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

# Run a specific number of iterations on a new research goal
python main.py --research_goal "Investigate the role of mitochondrial dysfunction in neurodegenerative diseases" --run 3

# List all existing research goals
python main.py --list_goals

# Resume an existing research session by its ID
python main.py --resume_id 64fd67eb-05a2-4e32-8dcc-cc63170d66ab --run 2

# Use a specific configuration
python main.py --config custom_config
```

#### Interactive Mode Commands

When running in interactive mode, you can use the following commands:

##### Research & Session Commands
- `goal: <text>` - Set a new research goal
- `run` - Run 1 iteration
- `run <N>` - Run N iterations
- `state` - Print the current system state
- `overview` - Generate and print a research overview

##### Hypothesis Management
- `hypotheses` - List all hypotheses, sorted by rating
- `hypothesis:ID` - View detailed information about a specific hypothesis
- `add-hypothesis:<text>` - Add your own hypothesis
- `feedback:ID <text>` - Provide feedback on a specific hypothesis
- `evolve:ID` - Request evolution of a specific hypothesis

##### Research Focus & Resources
- `focus: <text>` - Add a research focus area to guide exploration
- `focus-areas` - List all active research focus areas
- `resource: <url/text>` - Add a resource (paper, URL, description)
- `feedback` - List all feedback provided

##### Protocol Management
- `protocols` - List all experimental protocols
- `protocol:ID` - Generate a protocol for a specific hypothesis
- `generate-protocol` - Generate a protocol for a top hypothesis

##### Knowledge Search & Synthesis
- `search: <query>` - Search scientific databases across domains
- `synthesize: <query>` - Synthesize knowledge across scientific domains

##### System
- `help` - Show help information
- `exit` or `quit` - Exit the program

Example session:
```
> goal: Investigate the role of mitochondrial dysfunction in neurodegenerative diseases
Analyzing research goal: Investigating the role of mitochondrial dysfunction in neurodegenerative...
Detected relevant domains: biomedicine, biology, chemistry
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

> search: mitochondrial ROS in Alzheimer's
Searching scientific databases for: mitochondrial ROS in Alzheimer's
Detected relevant domains: biomedicine, biology
Searching across domains...

============ SEARCH RESULTS ============

BIOMEDICINE DOMAIN:
  1. Mitochondria-derived reactive oxygen species and Alzheimer's disease
     Authors: Swerdlow, R.H., Burns, J.M., Khan, S.M.
     Journal: Biochimica et Biophysica Acta
     Year: 2014
     URL: https://pubmed.ncbi.nlm.nih.gov/24189435/

  2. Mitochondrial dysfunction and oxidative stress in Alzheimer's disease
     Authors: Lin, M.T., Beal, M.F.
     Journal: Nature
     Year: 2006
     URL: https://pubmed.ncbi.nlm.nih.gov/17051205/
  ...

> synthesize: relationship between mitophagy and neurodegeneration
Synthesizing knowledge across domains for: relationship between mitophagy and neurodegeneration
Detected relevant domains: biomedicine, biology
Gathering and synthesizing information across domains...

# Knowledge Synthesis for: relationship between mitophagy and neurodegeneration

## Domains Overview
- Biomedicine: Relevance score 0.86
- Biology: Relevance score 0.72
- Chemistry: Relevance score 0.35

## Biomedicine Domain Findings
### 1. Impaired Mitophagy in Parkinson's Disease
PINK1/Parkin-mediated mitophagy is crucial for removing damaged mitochondria, and defects in this pathway are implicated in Parkinson's disease pathogenesis. Mutations in PINK1 and Parkin genes disrupt mitophagy, leading to accumulation of dysfunctional mitochondria...
Source: pubmed

#### Cross-domain connections:
- Biology: Mitochondrial quality control mechanisms, Autophagy receptor proteins

## Biology Domain Findings
### 1. Mitophagy and Mitochondrial Dynamics
Mitophagy is coordinated with mitochondrial fusion and fission processes. Fission segregates damaged mitochondrial components, making them accessible for mitophagy, while fusion helps redistribute mitochondrial contents to maintain functional networks...
Source: uniprot

## Synthesis Across Domains
The findings suggest connections between:
- Biomedicine and Biology: mitochondrial dysfunction, oxidative stress, autophagy
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

## System Functionality

### How the Watson Co-Scientist Works

The Watson Co-Scientist follows a systematic approach to scientific hypothesis generation and evaluation:

1. **Research Goal Specification**
   - The process begins with a research goal specified by a scientist in natural language
   - The system parses this goal, identifying key concepts, domains, and constraints
   - A research plan configuration is created to guide the hypothesis generation process

2. **Hypothesis Generation**
   - The Generation Agent produces initial hypotheses using several methods:
     - Literature exploration via scientific database integration
     - Simulated scientific debates among experts
     - Iterative identification of testable assumptions
     - Expansion into previously unexplored research areas
   - Each hypothesis includes: title, summary, detailed description, and supporting evidence

3. **Multi-level Hypothesis Review**
   - The Reflection Agent conducts thorough reviews of each hypothesis:
     - **Initial Review**: Quick assessment of correctness, quality, and novelty
     - **Full Review**: In-depth analysis with literature grounding
     - **Deep Verification Review**: Decomposition into constituent assumptions with confidence levels
     - **Observation Review**: Assessment of whether hypothesis explains existing experimental findings
     - **Simulation Review**: Step-by-step modeling of the hypothesized mechanism

4. **Tournament-based Ranking**
   - The Ranking Agent conducts an Elo-based tournament to evaluate hypotheses:
     - Pairwise comparisons through multi-turn scientific debates
     - Focus on novelty, correctness, and testability
     - Optimization using the Proximity Agent's graph to compare similar hypotheses
     - Progressive refinement of rankings across multiple iterations

5. **Hypothesis Evolution**
   - The Evolution Agent improves promising hypotheses using multiple strategies:
     - **Standard Improvement**: Addressing weaknesses identified in reviews
     - **Domain Knowledge Integration**: Grounding in specialized scientific knowledge
     - **Cross-domain Inspiration**: Applying concepts from other scientific fields
     - **Knowledge Graph Enhancement**: Using extracted scientific literature connections
     - **Knowledge Synthesis**: Consolidating information from multiple scientific sources
     - **Out-of-box Thinking**: Generating unconventional alternatives
     - **Simplification**: Improving clarity while preserving core ideas
     - **Combination**: Merging strengths of multiple hypotheses

6. **Scientific Knowledge Processing**
   - The system uses multiple approaches to process scientific knowledge:
     - **Multi-database Integration**: Searching across PubMed, ArXiv, etc.
     - **Domain Ontology Utilization**: Mapping concepts within specialized domains
     - **Cross-domain Synthesis**: Finding connections across different disciplines
     - **Paper Knowledge Extraction**: Processing full papers beyond abstracts
     - **Knowledge Graph Construction**: Building a graph of entities and relationships

7. **Advanced Scientific Paper Processing**
   - The Paper Knowledge Extraction System enables comprehensive literature understanding:
     - **Multi-Source PDF Retrieval**: Downloads papers from ArXiv, PubMed, Nature, Science, etc. with specialized handlers for each source
     - **Structured Content Extraction**: Processes text, figures, tables, equations, and citations with hierarchical section organization
     - **LLM-Powered Knowledge Extraction**: Uses large language models to identify key scientific entities, relationships, findings, and methodologies
     - **Dynamic Knowledge Graph Construction**: Builds and maintains a graph of scientific concepts and their relationships across multiple papers
     - **Entity Deduplication**: Identifies and merges similar scientific concepts using semantic similarity
     - **Knowledge-Enhanced Evolution**: Leverages paper knowledge graphs to improve hypothesis quality with literature-grounded insights
     - **Path Finding**: Discovers novel connections between scientific concepts that may not be explicitly stated in any single paper

8. **Research Synthesis and Reporting**
   - The Meta-review Agent synthesizes results for the scientist:
     - Identifies patterns across reviews and debates
     - Summarizes top-ranked hypotheses into a research overview
     - Outlines potential research areas with justifications
     - Suggests specific experiments and testable predictions
     - Identifies potential domain experts for collaboration

9. **Scientist-in-the-loop Interaction**
   - Throughout the process, scientists can:
     - Refine the research goal based on generated hypotheses
     - Provide manual reviews of system-generated hypotheses
     - Contribute their own hypotheses to the tournament
     - Direct the system to explore specific research directions
     - Resume sessions across multiple research periods
     - Pick up where they left off with all context preserved
     - Integrate their expertise with the system's capabilities

### Example System Workflow

1. Scientist specifies: "Investigate the molecular mechanisms of protein misfolding in neurodegenerative diseases"
2. System detects relevant domains: biomedicine, biochemistry, molecular biology
3. Generation Agent creates initial hypotheses about protein misfolding mechanisms
4. Reflection Agent reviews hypotheses for scientific validity and novelty
5. Tournament ranks hypotheses through simulated scientific debates
6. Evolution Agent improves promising hypotheses using specialized knowledge
7. Paper Knowledge Extraction System processes relevant scientific papers
8. Knowledge Graph connects concepts across multiple papers and domains
9. Meta-review Agent synthesizes findings into a comprehensive research overview
10. Scientist reviews the output, provides feedback, and guides further exploration

## License

MIT