# mini-RAUL: Research Co-Scientist

A minimalistic multi-agent system for scientific research assistance.

## Overview

mini-RAUL is a command-line tool that helps scientists explore research hypotheses through a multi-agent system. The system generates, evaluates, ranks, and evolves scientific hypotheses based on a research goal provided by the user.

## Features

- Multiple specialized AI agents working together
- Generation of diverse research hypotheses
- Tournament-style ranking of hypotheses
- Detailed reflection and analysis
- Evolution and improvement of hypotheses
- Session-based workflow with user feedback
- Simple command-line interface

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd raul-co-scientist

# Install the package
cd mini_raul
pip install -e .
```

## Usage

### Creating a Session

```bash
mini-raul create-session my_research_session
```

### Starting Research

```bash
mini-raul start-research my_research_session "Develop a novel hypothesis for the key factor which causes ALS related to phosphorylation of a Nuclear Pore Complex nucleoporin." --preferences "Focus on providing a novel hypothesis with detailed explanation of the mechanism of action."
```

### Viewing Session Status

```bash
mini-raul session-status my_research_session
```

### Viewing Hypotheses

```bash
mini-raul print-hypotheses my_research_session
```

### Viewing Rankings

```bash
mini-raul print-rankings my_research_session
```

### Continuing Research with Feedback

```bash
mini-raul continue-research my_research_session --feedback "The hypothesis about TDP-43 is interesting, but I would like to see more focus on the role of oxidative stress."
```

### Getting a Meta-Review

```bash
mini-raul get-meta-review my_research_session
```

## System Architecture

The system consists of the following components:

1. **Core**
   - Session management
   - Coordinator for orchestrating agents

2. **Agents**
   - Generation agents (literature exploration, scientific debate)
   - Ranking agents (tournament-style ranking)
   - Reflection agents (full review, tournament review, deep verification)
   - Evolution agents (improvement, simplification, extension)
   - Meta-review agents (research overview)

3. **Models**
   - LLM clients for different providers (Azure OpenAI, OpenAI, Ollama)

4. **CLI**
   - Command-line interface for interacting with the system

## Development

This is an iterative project, starting with a basic framework and gradually adding more features. The current version focuses on the core multi-agent system without external tools like web search or code execution.

## License

[MIT License](LICENSE) 