# VulnGraph

**Paper Title:** Graph-Augmented Language Models for Software Vulnerability Detection

VulnGraph is a research project exploring the integration of graph neural networks (GNNs) and large language models (LLMs) for vulnerability detection in Java code. The system represents Java code as Abstract Syntax Trees (ASTs) and Control Flow Graphs (CFGs) using [PROGEX](https://github.com/ghaffarian/progex), learns graph embeddings, and augments them with LLM-based reasoning for detecting subtle semantic vulnerabilities.

---

## Features

- Parse Java code into AST/CFG graphs using PROGEX
- Generate graph embeddings with GNNs (GCN/Node2Vec)
- Integrate structural embeddings with LLM reasoning for vulnerability detection
- Benchmark multiple LLMs:
  - DeepSeek-Coder 1.3B
  - Qwen2.5-Coder 3B
  - StableCoder-3B
  - Phi-3.5-mini (3.5B)
- Evaluate on Java vulnerability datasets

---

## Tentative Project Structure

```
VulnGraph/
│── data/ # Datasets (Java vulnerable & safe code samples)
│ ├── raw/ # Raw Java source files
│ ├── graphs/ # PROGEX-extracted program graphs
│ └── embeddings/ # Graph/LLM embeddings ready for model input
│
│── src/ # Core source code
│ ├── progex_parser/ # Scripts for invoking PROGEX and exporting graphs
│ ├── graphs/ # GNN scripts
│ ├── llms/ # LLMs scripts
│ ├── graph_llm_integration/ # Hybrid GNN + LLM pipeline
│ └── training/ # Training scripts and evaluation
│
│── notebooks/ # Jupyter notebooks for experiments
│── requirements.txt # Python dependencies
│── LICENSE # MIT License
└── README.md # Project documentation
```

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/JugalGajjar/VulnGraph.git
cd VulnGraph
```

### 2. Create a Python environment

Using **conda**:

```bash
conda create -n vulngraph python=3.12 -y
conda activate vulngraph
```

Or with **venv**:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Datasets

## Datasets

Our work utilizes more than **35,000 Java code samples**, consisting of both safe and vulnerable code files. These were collected and curated from multiple sources, as shown below:

| Dataset                                                          | Code Files |
| ---------------------------------------------------------------- | ---------- |
| Vul4J                                                            | 263        |
| Kaggle Vulnerability Fixes Dataset                               | 35,000     |
| Kaggle CVE Fixes (Java code files extracted)                     | 1,162      |
| Synthetic Java Code Files (all safe, to prevent class imbalance) | 6,000      |
| **Total (before cleaning)**                                      | **42,425** |

---

### After Cleaning

After cleaning and preprocessing the code samples, we obtained a total of **35,610 error-free, CFG-executable Java code files**, consisting of:

- **29,376 vulnerable files**
- **6,234 safe files**

This final cleaned dataset was curated from the three original sources along with additional synthetic safe code files. The inclusion of synthetic safe samples helps prevent class imbalance and ensures that the model does not overfit to vulnerable patterns alone.

---

## Models

- **Graph Embeddings**: GCN / Node2Vec
- **LLMs**: DeepSeek-Coder 1.3B, Qwen2.5-Coder 3B, StableCoder-3B, Phi-3.5-mini
- **Hybrid Model**: Concatenation of graph embeddings `h_G` with LLM semantic embeddings `h_llm` for classification

---

## Running the Pipeline

---

## Paper

This repository supports the research paper:  
**"Graph-Augmented Language Models for Software Vulnerability Detection"**  
Contribution: Hybridizing structural program graphs and semantic code reasoning for accurate vulnerability detection in Java code files.

---

## License

MIT License – free to use and modify with attribution.

---

## Authors

- Jugal Gajjar
- Kaustik Ranaware
<<<<<<< HEAD
- Kamalasankari Subramaniakuppusamy
- Relsy Puthal
=======
- Kamalasankari Subramaniakuppusamy
>>>>>>> 108751145ee404df8c242111436b2a016d7034af
