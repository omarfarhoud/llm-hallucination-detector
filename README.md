# LLM Hallucination Detector

Multi-signal hallucination detection system for LLM outputs.

## Features

- Semantic similarity checking
- LLM-as-judge evaluation
- Citation verification
- Confidence-aware aggregation

## Run with Docker
```bash
docker build -t hallucination-detector:latest .
docker run -p 8000:8000 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 hallucination-detector:latest
```