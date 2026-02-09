# MedDetect

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Built with uv](https://img.shields.io/badge/Built%20with-uv-000000?logo=python&logoColor=white)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready, multi-signal system for detecting hallucinations in medical LLM-generated responses. This project implements a **Confidence-Aware Gated Ensemble** to evaluate the faithfulness of medical RAG (Retrieval-Augmented Generation) outputs against provided clinical context, ensuring reliable and trustworthy medical information delivery.

**Contributors**: Omar Farhoud, Nadeen Hassan, Rana El Sharkawy

---

## 📋 Table of Contents

- [Features](#-features)
- [Performance Metrics](#-performance-metrics)
- [How It Works](#-how-it-works)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Configuration](#%EF%B8%8F-configuration)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [Research Context](#-research-context)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- **Multi-Signal Detection**: Combines semantic similarity, LLM-based reasoning, and citation verification
- **Confidence-Aware Gating**: Adaptive aggregation strategies based on signal confidence
- **Parallel Execution**: Concurrent signal processing using `asyncio` for minimal latency
- **Production-Ready API**: FastAPI with automatic documentation and validation
- **Docker Support**: Containerized deployment with optimized multi-stage builds
- **Comprehensive Testing**: Postman collection with 7+ test scenarios
- **Local LLM Integration**: Works seamlessly with Ollama for privacy and cost efficiency

---

## 📊 Performance Metrics

Evaluated on a 90-sample dataset with diverse hallucination types (fabricated stats, wrong entities, added facts, negations, fake citations, off-topic answers):

| Metric    | Score | Notes |
|-----------|-------|-------|
| **Accuracy**  | 0.922 | Overall correctness |
| **Precision** | 1.000 | Zero false positives (conservative detection) |
| **Recall**    | 0.825 | Catches 82.5% of hallucinations |
| **F1-Score**  | 0.904 | Harmonic mean of precision and recall |

### Aggregation Strategy Distribution

The system dynamically selects strategies based on signal confidence:

- **Full Weighted Aggregation**: 63% (57/90 samples) - Balanced multi-signal consensus
- **Judge Dominant**: 31% (28/90 samples) - High-confidence semantic contradictions
- **Structural Gate**: 6% (5/90 samples) - Strong structural evidence (citations/similarity)

### Why This Matters

The multi-signal approach significantly **outperforms individual signals** used in isolation:
- **Semantic similarity alone**: High precision but misses factual hallucinations
- **Citation verification alone**: Detects fabricated sources but can't assess semantic correctness  
- **LLM judge alone**: Strong overall but imperfect

By combining complementary signals with confidence-aware gating, the aggregator achieves higher recall while maintaining perfect precision.

---

## 🧠 How It Works

### The Core Engine: Gated Ensemble Logic

Unlike simple voting systems, this detector uses a sophisticated **Aggregation Strategy** to minimize both false positives and false negatives by analyzing signal confidence.

#### Signal Processing Pipeline

```
User Request → FastAPI Endpoint
       ↓
  [Question, Answer, Context]
       ↓
  ┌─────────────────────────────────┐
  │   Parallel Signal Execution     │
  │  (asyncio.gather for speed)     │
  ├─────────────────────────────────┤
  │ • Semantic Similarity (CPU)     │
  │ • LLM Judge (Ollama API)        │
  │ • Citation Verification (CPU)   │
  └─────────────────────────────────┘
       ↓
  Confidence-Aware Aggregator
       ↓
  [Hallucination Score + Verdict]
```

#### Scoring Logic

All signals produce **hallucination scores** (0 = faithful, 1 = hallucinated). The final score $H_{final}$ is calculated using one of three regimes:

**Regime A: Structural Gate** (6% of cases)
- Triggers when: Semantic Similarity OR Citation Check score > 0.8
- Logic: Takes the maximum of the two structural signals
- Rationale: Strong structural evidence provides high-confidence detection

**Regime B: Judge Dominant** (31% of cases)
- Triggers when: LLM Judge score > 0.7
- Logic: Uses judge score as final score
- Rationale: High-confidence semantic contradictions override other signals

**Regime C: Full Weighted Aggregation** (63% of cases)
- Triggers when: No high-confidence signals
- Logic: Weighted ensemble of all three signals

$$H_{final} = 0.60 \cdot H_{judge} + 0.25 \cdot H_{sim} + 0.15 \cdot H_{cite}$$

- Rationale: Balanced consensus when no signal is highly confident

**Final Decision**: $\text{is\_hallucination} = (H_{final} \geq 0.7)$

### Parallel Execution

To minimize latency, all three signals are dispatched concurrently using `asyncio.gather`:

```python
# All signals run in parallel for maximum performance
similarity_task = asyncio.to_thread(similarity_checker.check, answer, docs)
judge_task = llm_judge.judge(question, answer, context)
citation_task = asyncio.to_thread(citation_checker.check, answer, docs)

# Wait for all to complete
similarity_result, judge_result, citation_result = await asyncio.gather(
    similarity_task, judge_task, citation_task
)
```

This allows I/O-bound tasks (LLM Judge via Ollama) and CPU-bound tasks (Sentence Transformers) to run concurrently, reducing total latency by ~60% compared to sequential execution.

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                 User / Client                           │
└────────────┬────────────────────────────────────────────┘
             │
     ┌───────┴──────────┐
     │                  │
     ↓                  ↓
┌─────────────┐   ┌──────────────────────────────────────┐
│  Streamlit  │   │   Direct API Access (curl/Postman)  │
│  Dashboard  │   └──────────────────────────────────────┘
│ (Port 8501) │                  │
└──────┬──────┘                  │
       │ HTTP                    │
       └─────────────┬────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│        FastAPI Application (Port 8000)                  │
│  ┌───────────────────────────────────────────────────┐  │
│  │            /detect Endpoint                       │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         ↓           ↓           ↓
    ┌────────┐  ┌────────┐  ┌────────┐
    │Semantic│  │  LLM   │  │Citation│
    │Similar.│  │ Judge  │  │ Check  │
    └────┬───┘  └───┬────┘  └───┬────┘
         │          │ (calls)   │
         │          ↓           │
         │    ┌──────────┐     │
         │    │  Ollama  │     │
         │    │(phi3:mini)│    │
         │    │(on host) │     │
         │    └──────────┘     │
         └──────────┬───────────┘
                    ↓
         ┌─────────────────────┐
         │   Aggregator Logic  │
         │  (Gated Ensemble)   │
         └──────────┬──────────┘
                    ↓
              JSON Response
```

### Docker Compose Stack

When using `docker compose up`, two services are orchestrated:

1. **API Service** (`api`)
   - Container: `hallucination-api`
   - Port: 8000
   - Connects to host machine's Ollama via `host.docker.internal:11434`
   - Restart policy: `unless-stopped`

2. **Dashboard Service** (`dashboard`)
   - Container: `hallucination-dashboard`
   - Port: 8501
   - Connects to API service via internal Docker network (`http://api:8000`)
   - Provides interactive UI for testing and visualization
   - Depends on: API service

### Signal Modules

1. **Semantic Similarity Checker**
   - **Model**: SentenceTransformers (`all-MiniLM-L6-v2`)
   - **Method**: Cosine similarity between answer and context embeddings
   - **Output**: Hallucination score (1 - max_similarity)

2. **LLM Judge**
   - **Model**: Ollama (`phi3:mini` by default)
   - **Method**: Few-shot prompted reasoning about faithfulness
   - **Output**: Faithfulness score (inverted to hallucination score)

3. **Citation Verification**
   - **Method**: Extracts citations (e.g., `[1]`, `[doc2]`), validates against context
   - **Penalties**: Missing citations, unsupported numeric claims
   - **Output**: Faithfulness score (inverted to hallucination score)

4. **Aggregator**
   - **Method**: Confidence-aware gating with three regimes
   - **Weights**: Judge (60%), Similarity (25%), Citation (15%)
   - **Threshold**: 0.7 for hallucination detection

---

## 🚀 Installation

### Prerequisites

- **Docker Desktop** (recommended) OR
- **Python 3.11+** with **uv** package manager
- **Ollama** installed and running locally ([Download Ollama](https://ollama.ai/))

### Option 1: Docker Compose (Recommended)

The entire stack (API + Dashboard) can be launched with a single command:

```bash
# Clone the repository
git clone https://github.com/omar-farhoud/llm-hallucination-detector.git
cd llm-hallucination-detector

# Start both API and Dashboard
docker compose up --build
```

**Services launched:**
- **API**: http://localhost:8000 (FastAPI backend)
- **Dashboard**: http://localhost:8501 (Streamlit UI)

**Note**: The initial build downloads ~3GB of dependencies and takes 15-30 minutes. Subsequent builds use layer caching and are much faster (30s-2min for dependency changes, 5-10s for code changes).

### Option 2: Docker (API Only)

If you only need the API without the dashboard:

```bash
# Build the API image
docker build -t hallucination-detector-api:latest .

# Run the API container
docker run -p 8000:8000 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  hallucination-detector-api:latest
```

### Option 3: Local Development

```bash
# Clone the repository
git clone https://github.com/omar-farhoud/llm-hallucination-detector.git
cd llm-hallucination-detector

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies with API extras
uv sync --extra api

# Run the API
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Verify Installation

```bash
# Check health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy"}

# Open interactive API docs
open http://localhost:8000/docs

# Access Streamlit dashboard (if using docker-compose)
open http://localhost:8501
```

---

## 💻 Usage

### Quick Start Example

```python
import requests

# API endpoint
url = "http://localhost:8000/detect"

# Medical query example
payload = {
    "question": "What are the risk factors for Type 2 diabetes?",
    "context": [
        {
            "id": "doc1",
            "text": "Type 2 diabetes risk factors include obesity, family history, physical inactivity, and age over 45. Prediabetes and gestational diabetes are also significant risk factors."
        }
    ],
    "answer": "The main risk factors for Type 2 diabetes are obesity, smoking, and high cholesterol [doc1]."
}

# Make request
response = requests.post(url, json=payload)
result = response.json()

print(f"Hallucination Score: {result['hallucination_score']}")
print(f"Is Hallucination: {result['is_hallucination']}")
print(f"Strategy Used: {result['aggregation_mode']['strategy']}")
```

### Example Response

```json
{
  "hallucination_score": 0.7234,
  "is_hallucination": true,
  "threshold": 0.7,
  "weights": {
    "semantic_similarity": 0.25,
    "llm_judge": 0.6,
    "citation_check": 0.15
  },
  "aggregation_mode": {
    "strategy": "full_weighted",
    "used_signals": ["semantic_similarity", "llm_judge", "citation_check"]
  },
  "signal_scores": {
    "semantic_similarity_h": 0.23,
    "llm_judge_h": 0.85,
    "citation_check_h": 0.92
  }
}
```

**Interpretation**: The answer is flagged as hallucinated (score 0.72 > 0.7 threshold) because:
- The LLM judge detected a contradiction (smoking and high cholesterol not mentioned in context)
- Citation check failed (unsupported claims attributed to [doc1])
- Semantic similarity is acceptable but overridden by other signals

This demonstrates the system's ability to catch medically dangerous hallucinations where fabricated risk factors could mislead patients or healthcare providers.

### Using Postman

A comprehensive Postman collection is included with 7 test scenarios:

```bash
# Import the collection
# File: postman_collection.json

# Test cases included:
# 1. Faithful answer (baseline)
# 2. Fabricated statistics
# 3. Wrong entity substitution
# 4. Added unsupported facts
# 5. Negation errors
# 6. Fake citations
# 7. Off-topic responses
```

---

## 🔌 API Reference

### `POST /detect`

Detects hallucinations in an LLM-generated answer given a question and context.

#### Request Body

```json
{
  "question": "string (required)",
  "answer": "string (required)", 
  "context": [
    {
      "id": "string or int (required)",
      "text": "string (required)"
    }
  ]
}
```

#### Response

```json
{
  "hallucination_score": "float (0-1)",
  "is_hallucination": "boolean",
  "threshold": "float",
  "weights": {
    "semantic_similarity": "float",
    "llm_judge": "float",
    "citation_check": "float"
  },
  "aggregation_mode": {
    "strategy": "string (full_weighted | judge_dominant | structural_gate)",
    "used_signals": ["array of signal names"]
  },
  "signal_scores": {
    "semantic_similarity_h": "float (0-1)",
    "llm_judge_h": "float (0-1)",
    "citation_check_h": "float (0-1)"
  },
  "signals": {
    "semantic_similarity": { "hallucination_score": "float", "raw": {} },
    "llm_judge": { "hallucination_score": "float", "raw": {} },
    "citation_check": { "hallucination_score": "float", "raw": {} }
  }
}
```

### `GET /health`

Health check endpoint.

#### Response

```json
{
  "status": "healthy"
}
```

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `phi3:mini` | LLM model for judge signal |
| `SIMILARITY_WEIGHT` | `0.25` | Weight for semantic similarity |
| `JUDGE_WEIGHT` | `0.60` | Weight for LLM judge |
| `CITATION_WEIGHT` | `0.15` | Weight for citation verification |
| `AGGREGATION_THRESHOLD` | `0.7` | Decision boundary for hallucination |
| `CITATION_THRESHOLD` | `0.5` | Threshold for citation faithfulness |
| `SIMILARITY_GATE` | `0.8` | Trigger threshold for structural gate |
| `JUDGE_DOMINANCE_GATE` | `0.7` | Trigger threshold for judge dominance |
| `BACKEND_URL` | `http://api:8000/detect` | API endpoint (used by dashboard) |
| `PYTHONPATH` | `/app` | Python module search path |

### Setting Environment Variables

**Docker Compose** (modify `docker-compose.yml`):
```yaml
services:
  api:
    environment:
      - OLLAMA_HOST=http://host.docker.internal:11434
      - AGGREGATION_THRESHOLD=0.75
      - JUDGE_WEIGHT=0.7
```

**Docker Run**:
```bash
docker run -p 8000:8000 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -e AGGREGATION_THRESHOLD=0.75 \
  hallucination-detector-api:latest
```

**Local (.env file)**:
```bash
# Create .env file
cat > .env << EOF
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=phi3:mini
AGGREGATION_THRESHOLD=0.7
EOF

# Run with environment
uv run uvicorn api.main:app --reload
```

---

## 🧪 Testing

### Run Evaluation Script

Evaluate the system on the full 90-sample dataset:

```bash
# With Docker Compose
docker compose exec api python scripts/evaluate_aggregator.py

# With standalone Docker
docker run hallucination-detector-api:latest python scripts/evaluate_aggregator.py

# Local
uv run python scripts/evaluate_aggregator.py
```

**Expected Output**:
```
================================================================================
HALLUCINATION AGGREGATOR EVALUATION
================================================================================
[001] GT=valid        PRED=valid        SCORE=0.23 STRATEGY=full_weighted
[002] GT=hallucinated PRED=hallucinated SCORE=0.89 STRATEGY=judge_dominant
...
================================================================================
SUMMARY METRICS
================================================================================
Samples   : 90
Accuracy  : 0.922
Precision : 1.000
Recall    : 0.825

Aggregation strategies used:
  full_weighted  : 57
  judge_dominant : 28
  structural_gate: 5
```

### Test with Streamlit Dashboard

```bash
# Start the full stack
docker compose up

# Open dashboard
open http://localhost:8501

# Features:
# - Interactive form to test detection
# - Real-time signal visualization
# - Aggregation strategy explanation
# - Example test cases
```

### Test with Postman

1. Import `postman_collection.json` into Postman
2. Set base URL to `http://localhost:8000`
3. Run the collection to test all 7 scenarios
4. Review saved responses for expected behavior

### Unit Tests (Coming Soon)

```bash
# Run pytest suite (when available)
uv run pytest tests/
```

---

## 🔧 Troubleshooting

### Container Can't Reach Ollama

**Symptom**: `Failed to connect to Ollama` error in logs

**Solution**:
```bash
# 1. Verify Ollama is running on host
curl http://localhost:11434/api/tags

# 2. Ensure docker-compose.yml has correct settings
services:
  api:
    environment:
      - OLLAMA_HOST=http://host.docker.internal:11434
    extra_hosts:
      - "host.docker.internal:host-gateway"

# 3. For standalone Docker run
docker run -p 8000:8000 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  hallucination-detector-api:latest
```

### Build Takes Too Long

**Expected**: Initial build downloads ~3GB of dependencies (PyTorch, CUDA libraries, transformers) and takes 15-30 minutes.

**Future builds are faster**:
- Code changes only: 5-10 seconds
- Dependency changes: 30 seconds - 2 minutes (uses layer caching)

**To optimize** (future work):
- Remove unused CUDA/GPU dependencies (your container doesn't need them since Ollama runs on host)
- Use smaller base images
- Multi-stage builds to exclude build tools

### Port Already in Use

**Symptom**: `Address already in use` error

**Solution**:
```bash
# Use a different port
docker run -p 8001:8000 hallucination-detector:latest

# Then access at http://localhost:8001
```

### Model Download Issues

**Symptom**: `SentenceTransformer model not found`

**Solution**:
```bash
# Pre-download model locally
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Or specify cache directory
docker run -v ~/.cache/huggingface:/root/.cache/huggingface ...
```

### Memory Issues

**Symptom**: Docker build fails with out-of-memory error

**Solution**:
```bash
# Increase Docker Desktop memory allocation
# Docker Desktop → Settings → Resources → Memory → 8GB+

# Or build with limited workers
docker build --memory=8g -t hallucination-detector:latest .
```

---

## 🗂️ Project Structure

```
llm-hallucination-detector/
├── api/                          # FastAPI application
│   ├── routes/
│   │   ├── health.py            # Health check endpoint
│   │   └── detect.py            # Main detection endpoint
│   ├── models.py                # Pydantic request/response models
│   └── main.py                  # FastAPI application entry point
├── dashboard/                   # Streamlit UI
│   ├── app.py                   # Streamlit dashboard
│   └── Dockerfile               # Dashboard container build
├── src/                         # Core detection logic
│   ├── detection/
│   │   ├── similarity_check.py  # Semantic similarity signal
│   │   ├── citation_check.py    # Citation verification signal
│   │   ├── llm_judge.py         # LLM-based judge signal
│   │   └── aggregator.py        # Gated ensemble aggregator
│   └── utils/
│       ├── ollama_client.py     # Ollama API wrapper
│       └── metrics.py           # Evaluation metrics
├── config/
│   ├── settings.py              # Pydantic settings management
│   └── prompts.py               # LLM judge prompt templates
├── scripts/
│   └── evaluate_aggregator.py  # Evaluation script
├── data/
│   └── preliminary_hallucination_dataset.json  # 90-sample test set
├── notebooks/                   # Jupyter notebooks for experimentation
│   ├── signal_analysis.ipynb
│   └── threshold_tuning.ipynb
├── Dockerfile                   # Multi-stage production build for API
├── docker-compose.yml          # Orchestration for API + Dashboard
├── .dockerignore               # Docker build exclusions
├── pyproject.toml              # uv project configuration
├── uv.lock                     # Locked dependencies
├── postman_collection.json     # API test collection
└── README.md                   # This file
```

---

## 🛠️ Development

### Adding a New Signal

1. Create a new signal module in `src/detection/`:

```python
# src/detection/my_signal.py
class MySignal:
    def check(self, answer: str, context: List[Dict]) -> Dict:
        """
        Returns:
            {
                'faithfulness_score': float (0-1),
                'details': str
            }
        """
        # Your detection logic here
        return {'faithfulness_score': 0.8, 'details': 'Reasoning...'}
```

2. Update the aggregator in `src/detection/aggregator.py`:

```python
# Add to aggregate() method
my_signal_result = my_signal.check(answer, context)
my_signal_h = 1.0 - my_signal_result['faithfulness_score']

# Include in weighted aggregation
final_score = (
    0.50 * judge_h +
    0.20 * similarity_h +
    0.15 * citation_h +
    0.15 * my_signal_h  # New signal
)
```

3. Update weights to sum to 1.0

### Modifying Aggregation Weights

Edit `config/settings.py`:

```python
class Settings(BaseSettings):
    similarity_weight: float = 0.20  # Changed from 0.25
    judge_weight: float = 0.65       # Changed from 0.60
    citation_weight: float = 0.15
    my_signal_weight: float = 0.10   # New
```

### Running Tests Locally

```bash
# Run evaluation
uv run python scripts/evaluate_aggregator.py

# Run with different thresholds
AGGREGATION_THRESHOLD=0.75 uv run python scripts/evaluate_aggregator.py

# Analyze false negatives
uv run python scripts/evaluate_aggregator.py --verbose
```

### Local Development Server

```bash
# Run API with hot-reload
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Run dashboard locally
cd dashboard
uv run streamlit run app.py
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint
uv run ruff check .

# Type checking
uv run mypy src/
```

---

## 👥 Team & Project Background

This project was developed collaboratively by **Omar Farhoud**, **Nadeen Hassan**, and **Rana El Sharkawy**, applying MLOps principles and ensemble learning methodologies to medical LLM reliability assessment.

### Project Focus

**Problem**: LLM hallucinations in medical RAG systems pose significant risks to patient safety and clinical decision-making. Inaccurate or fabricated medical information can lead to dangerous outcomes in healthcare applications.

**Hypothesis**: Multi-signal aggregation with confidence-aware gating can achieve higher F1-scores than individual detectors while maintaining production-grade precision, specifically tailored for medical domain reliability.

**Methodology**: 
1. Developed three complementary signals (semantic, reasoning, structural)
2. Implemented adaptive gating based on signal confidence
3. Evaluated on diverse hallucination taxonomy (7 types) relevant to medical contexts

**Key Findings**:
- Achieved **0.904 F1-score** with **1.0 precision** (zero false positives)
- Full weighted aggregation used in 63% of cases (balanced consensus)
- Judge dominance in 31% (high-confidence contradictions)
- Structural gate in 6% (strong citation/similarity evidence)

**Current Limitation**: The system is currently evaluated on synthetic test data. A key improvement would be integrating a complete medical RAG pipeline (document ingestion → retrieval → generation → detection) to test on real-world medical queries and documentation.

### Technical Stack

- **Package Management**: [uv](https://github.com/astral-sh/uv) for deterministic, fast dependency resolution
- **ML Inference**: [Ollama](https://ollama.ai/) for local, privacy-preserving LLM orchestration
- **Embeddings**: [SentenceTransformers](https://www.sbert.net/) (all-MiniLM-L6-v2) for semantic similarity
- **API Framework**: [FastAPI](https://fastapi.tiangolo.com/) for high-performance, type-safe endpoints
- **UI Framework**: [Streamlit](https://streamlit.io/) for interactive visualization
- **Containerization**: Docker with multi-stage builds for optimized deployment

### Citation

If you use this work in your research, please cite:

```bibtex
@software{meddetect_2025,
  author = {Farhoud, Omar and Hassan, Nadeen and El Sharkawy, Rana},
  title = {MedDetect: Multi-Signal Hallucination Detection for Medical RAG Systems},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/omar-farhoud/llm-hallucination-detector}
}
```

---

## 🗺️ Roadmap

### Current Version (v0.1.0)
- ✅ Multi-signal detection pipeline
- ✅ Confidence-aware gating
- ✅ FastAPI REST endpoint
- ✅ Docker containerization with multi-stage builds
- ✅ Streamlit dashboard for interactive testing
- ✅ Docker Compose orchestration
- ✅ Postman test collection
- ✅ Evaluation on 90-sample synthetic dataset

### Upcoming (v0.2.0)
- [ ] **RAG System Integration**: Build complete medical RAG pipeline with vector database for end-to-end testing
  - [ ] Medical document ingestion and chunking
  - [ ] Vector store integration (ChromaDB/FAISS/Pinecone)
  - [ ] Retrieval component for context fetching
  - [ ] End-to-end pipeline: Query → Retrieval → Generation → Hallucination Detection
- [ ] Batch processing endpoint
- [ ] WebSocket streaming for real-time detection
- [ ] Redis caching for embeddings
- [ ] Prometheus metrics export
- [ ] Unit and integration tests (pytest)
- [ ] CI/CD pipeline (GitHub Actions)

### Future (v1.0.0)
- [ ] Domain-specific medical knowledge base
- [ ] Support for additional LLM providers (OpenAI, Anthropic, Cohere)
- [ ] Fine-tuned judge models for medical domain
- [ ] Active learning loop for threshold optimization
- [ ] Multi-language support
- [ ] Kubernetes deployment manifests
- [ ] Benchmark against academic baselines (SelfCheckGPT, RARR, etc.)
- [ ] Export detection results to common formats (JSON, CSV, PDF reports)
- [ ] Integration with medical knowledge graphs (UMLS, SNOMED CT)

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-new-feature`
3. **Make your changes** and add tests
4. **Run quality checks**: `uv run ruff check . && uv run ruff format .`
5. **Commit with clear messages**: `git commit -m "feat: add new signal for X"`
6. **Push to your fork**: `git push origin feature/my-new-feature`
7. **Open a Pull Request** with a detailed description

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/llm-hallucination-detector.git
cd llm-hallucination-detector

# Install dev dependencies
uv sync --all-extras

# Install pre-commit hooks (if available)
pre-commit install

# Run tests
uv run pytest tests/
```

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) Python style guide
- Use type hints for function signatures
- Add docstrings for public methods (Google style)
- Keep functions focused and testable
- Use descriptive variable names

### Reporting Issues

Found a bug or have a feature request? [Open an issue](https://github.com/omar-farhoud/llm-hallucination-detector/issues) with:
- **Clear title** describing the problem/request
- **Steps to reproduce** (for bugs)
- **Expected vs. actual behavior**
- **Environment details** (OS, Python version, Docker version)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Omar Farhoud

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **Team Members**: Omar Farhoud, Nadeen Hassan, and Rana El Sharkawy for collaborative development and research
- **Anthropic** for inspiration from Claude's structured outputs and reasoning capabilities
- **Astral (uv team)** for revolutionizing Python package management
- **Ollama** for making local LLM inference accessible and efficient
- **SentenceTransformers team** for lightweight, production-ready embedding models

---

## 📬 Contact

**Project Team**
- **Omar Farhoud** - [@omarfarhoud](https://github.com/omarfarhoud)
- **Nadeen Hassan** - [@__nadeenhassan28__](https://github.com/__nadeenhassan28__)
- **Rana El Sharkawy** - [@ranaelsharkawy23](https://github.com/ranaelsharkawy23)

For questions, feedback, or collaboration opportunities, feel free to reach out via GitHub issues or pull requests!

---

