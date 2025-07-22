# ICSE-2026-SUB

A multi-representational framework for smart contract analysis using contrastive learning. This pipeline utilizes CFG, DFG, CCFG, AST, and Bytecode representations to extract embeddings that are aligned and contrasted using a co-attention-driven contrastive model.

---

## 📁 Dataset

All raw Solidity smart contracts must be placed inside the `dataset/` folder. These files serve as input for the various static analysis modules.

---

## ⚙️ Preprocessing Pipeline

Each source code file in the dataset is passed through the following processing modules:

- `CFG_Processor.py` & `cfg-generator.py` – Extract Control Flow Graphs
- `DFG.py` – Extract Data Flow Graphs
- `CCFG.py` – Build Contextual Control Flow Graphs
- `ast-generator.py` & `process_ast_data.py` – Generate Abstract Syntax Trees
- `code_to_bytecode.py` – Decompile smart contracts into EVM bytecode

> ✅ These modules process **all files** present in the `dataset/` directory and do **not** rely on the CSV controller.

---

## 🧠 Embedding Extraction

The processed representations are passed through dedicated encoders using `gat_feature_extractor.py`, which outputs vector embeddings:

- `Embeddings-CFG/` – CFG embeddings
- `Embeddings-DFG/` – DFG embeddings
- `Embeddings-CCFG/` – CCFG embeddings
- `Embeddings-AST/` – AST embeddings

Bytecode embeddings are generated separately using a masked language model (`mlm.py`):

- `Embeddings-Bytecode/` – Bytecode-level embeddings

---

## 🔁 Contrastive Learning

All five embeddings are input to the contrastive learning pipeline defined in `Contrastive_Learning_model.py`, consisting of:

- **Co-Attention Layer** – Learns cross-view alignment between different representations
- **Contrastive Loss Module** – Encourages similarity across views of the same contract and dissimilarity between unrelated ones

The contrastive loss is also used to update the co-attention layer for better joint representation learning.

---

## 🧪 Classification (Evaluation)

To assess the quality of learned representations, run `embedding_classifier.py` which uses the contrastively learned embeddings for downstream classification tasks.

---

## 📄 CSV Controller

All major modules (except the initial raw processors) accept a CSV file as input. This file acts as the pipeline controller:

- Specifies which contract files to include
- Provides labels or metadata as needed

> Format of CSV:  
> ```
> File,Label
> contract_1.sol,1
> contract_2.sol,0
> ...
> ```

---

## 🚀 Getting Started

1. **Prepare the Dataset**
   
   Place all your Solidity files in the `dataset/` directory.
   Place your CSV file (in `File,Label` format) in dataset folder along Solidity files .
