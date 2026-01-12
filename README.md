# Sanskrit Document RAG System

**Author:** Rushabh Katekhaye
**Project Type:** Industry Assignment
**Execution Mode:** CPU-only, Fully Local

---

## 1. Project Overview

This project implements a **Sanskrit Document Question Answering system** using a Retrieval-Augmented approach under strict constraints:

* CPU-only execution
* Fully local (no external APIs or cloud inference)
* Support for Sanskrit (Devanagari script) and English queries

The objective was to explore how far modern NLP techniques can be practically applied to Sanskrit under real-world hardware limitations, and to design a system that is stable, reproducible, and explainable.

The system retrieves relevant Sanskrit text segments from documents and returns clean, reliable answers extracted from the source.

---

## 2. Problem Statement

Building an intelligent question answering system for Sanskrit is challenging because:

* Sanskrit has complex morphology and grammar.
* Most NLP models are trained primarily on English.
* Multilingual models often produce mixed-script or unstable outputs.
* CPU-only environments severely limit model size and inference capability.

The goal was to design a system that works reliably within these constraints.

---

## 3. System Architecture

The system consists of the following components:

1. **Document Loader**

   * Loads Sanskrit documents from `.txt` format.

2. **Text Chunking**

   * Splits documents into overlapping chunks to preserve context.

3. **Embedding Model**

   * Converts chunks into vector embeddings using a multilingual sentence transformer.

4. **Vector Store**

   * Stores embeddings in FAISS for similarity search.

5. **Query Processor**

   * Converts user query into vector and retrieves relevant chunks.

6. **Answer Module**

   * Uses extractive approach to present relevant Sanskrit text and English context.

---

## 4. Technologies Used

| Component        | Technology                       |
| ---------------- | -------------------------------- |
| Language         | Python 3                         |
| Vector Store     | FAISS                            |
| Embeddings       | sentence-transformers            |
| Models Tested    | IndicBART, ByT5, mT5, Phi, Gemma |
| Local LLM Runner | Ollama                           |
| Execution        | CPU only                         |
| Storage          | Local filesystem                 |

---

## 5. Models Evaluated

| Model             | Purpose                    | Result                        |
| ----------------- | -------------------------- | ----------------------------- |
| IndicBART         | Indic language generation  | Mixed-script outputs          |
| ByT5              | Character-level generation | Clean text but weak reasoning |
| mT5               | Multilingual generation    | Slow and inconsistent         |
| Phi (Ollama)      | Local LLM                  | Timeout and instability       |
| Gemma 2B (Ollama) | Local LLM                  | Heavy and unreliable on CPU   |

**Final Decision:**
Use **extractive retrieval + formatting** for reliability and accuracy.

---

## 6. Implementation Strategy

Instead of generating answers, the system:

* Retrieves the most relevant Sanskrit chunks using similarity search.
* Extracts and formats clean Sanskrit text.
* Provides English explanation when present.
* Displays similarity scores for transparency.

This avoids hallucination, encoding issues, and instability seen in generative models.

---

## 7. Results

| Metric                                | Result             |
| ------------------------------------- | ------------------ |
| Retrieval accuracy (Sanskrit queries) | ~75% similarity    |
| Retrieval accuracy (English queries)  | ~40–50% similarity |
| Response time                         | 2–3 seconds        |
| Stability                             | High               |
| Script correctness                    | Pure Devanagari    |

---

## 8. Challenges Faced

* Library compatibility issues (torch, transformers, huggingface_hub).
* Mixed-script outputs from multilingual models.
* Model instability on CPU.
* Timeout issues with Ollama models.
* Encoding inconsistencies in generated text.

All issues were documented and addressed through iterative testing.

---

## 9. Engineering Learnings

* CPU-only environments strongly limit generative NLP.
* Sanskrit requires specialized handling.
* Retrieval systems are more reliable than generation in constrained setups.
* Transparency and reproducibility are critical in production-style systems.

---

## 10. Limitations

* No abstractive generation of new answers.
* Dependent on quality of source documents.
* No semantic reasoning beyond retrieval.

---

## 11. Future Improvements

* GPU-based deployment for generation.
* Fine-tuning Sanskrit-specific models.
* Custom tokenizer for Devanagari.
* UI-based interface.

---

## 12. Engineering Transparency

This project was developed through iterative evaluation of multiple open-source models and architectures.

All decisions were made based on:

* Observed system performance,
* Stability and reproducibility,
* Output quality,
* Hardware constraints.

No proprietary APIs or cloud services were used.

---

## 13. How to Run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cd code
python main.py
```

---

## 14. Conclusion

This project demonstrates that reliable Sanskrit QA is achievable under CPU-only constraints using intelligent retrieval and formatting.

While generative models struggled, the final system offers stability, accuracy, and transparency — which are more important for practical engineering deployment.

---

