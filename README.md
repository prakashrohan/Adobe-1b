
# 🤖 Persona-Driven Document Intelligence

This project performs intelligent section extraction and summarization from PDF documents based on a given **persona** and **task** — entirely CPU-based. It uses spaCy’s GloVe embeddings (300-d) for semantic similarity, keyword scoring, and summarization. Ideal for document intelligence systems constrained by size, performance, or offline requirements.

---

## 🚀 Features

- ✅ Uses **spaCy**’s GloVe (`en_core_web_md`) – no PyTorch or transformers
- 🧠 Hybrid semantic scoring with cosine similarity, keyword overlap, and keyphrase matches
- 📄 PDF parsing with **pdfplumber**
- 🔖 Extracts meaningful page chunks using overlapping sentence segmentation
- 📝 Generates **section headlines** and **summaries** using enhanced **TextRank**
- 📦 Fully Dockerized and offline-capable
- ⚡ Efficient on CPU with <1GB memory usage

---

## 🧱 How It Works

1. **PDF Page Extraction**
   - Reads each page using `pdfplumber`
   - Cleans, strips footers/headers, filters low-quality pages

2. **Intelligent Page Filtering**
   - Ranks pages based on overlap with keywords/phrases from persona & task

3. **Text Chunking**
   - Breaks filtered pages into overlapping chunks (~400 words)

4. **Hybrid Chunk Scoring**
   - Computes similarity with persona+task embedding
   - Adds keyword & phrase match boost
   - Scores weighted into a `combined_score`

5. **Document Aggregation**
   - Ranks top 5 documents based on chunk scores
   - Extracts best page per document

6. **Headline + Summary Generation**
   - Extracts page-level summary using TextRank + clustering
   - Builds headlines based on keyphrase context relevance

---

## 📦 Installation

### ✅ Requirements
- Python 3.8+
- `spaCy`, `pdfplumber`, `scikit-learn`, `numpy`, `networkx`, `tqdm`

### Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### `requirements.txt`:
```
numpy
tqdm
pdfplumber
spacy
scikit-learn
networkx
```

---

## 🐳 Docker Setup

### 🔧 Build Docker Image
```bash
docker build --platform linux/amd64 -t
mysolutionname:somerandomidentifier
```

### ▶️ Run in Container
```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --
network none mysolutionname:somerandomidentifier
```

Ensure `input/input.json` contains:
```json
{
  "persona": { "role": "Data Scientist" },
  "job_to_be_done": { "task": "Summarize key findings" },
  "documents": [
    { "filename": "somefile.pdf" }
  ]
}
```

---

## 🖥️ Local Usage

```bash
python main.py input/input.json output/output.json
```

---

## 📂 Output Format

```json
{
  "metadata": {
    "input_documents": ["somefile.pdf"],
    "persona": "Data Scientist",
    "job_to_be_done": "Summarize key findings",
    "processing_timestamp": "2025-07-28T..."
  },
  "extracted_sections": [
    {
      "document": "somefile.pdf",
      "section_title": "Summary of Results: Key Findings",
      "importance_rank": 1,
      "page_number": 3
    }
  ],
  "subsection_analysis": [
    {
      "document": "somefile.pdf",
      "refined_text": "This study explored...",
      "page_number": 3
    }
  ]
}
```

---

## 🧠 About the Embeddings

- Uses `spaCy`'s `en_core_web_md` GloVe-based model (≈120MB)
- L2-normalized vectors, cosine similarity = dot product
- Efficient caching (LRU with hash of `text[:200]`)

---

## 🧹 Memory Management

- Uses `SmartEmbedderManager.cleanup()` to:
  - Release spaCy model
  - Clear embedding cache
  - Force garbage collection

---


