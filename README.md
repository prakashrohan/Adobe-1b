# 🧠 ADOBE-1B – Intelligent Recipe Document Extractor

This project processes PDF documents containing recipe ideas and extracts structured content using AI-powered embeddings and smart document parsing.

---

## 📁 Project Structure

- `input/` – Folder for input PDFs and `input.json`
- `output/` – Folder where `output.json` will be saved
- `main.py` – Entry point for processing
- `SmartEmbedderManager.py` – Embedding manager (uses SentenceTransformer)
- `requirements.txt` – Python dependencies
- `Dockerfile` – Docker build instructions
- `.dockerignore` – Files ignored during Docker build
- `README.md` – You're here!

---

## 🚀 How to Run (with Docker)

> Make sure Docker is installed and running.

### Step 1: Place your input files

Put all `.pdf` files and a valid `input.json` file inside the `input/` directory.

Example:

- `input/Breakfast Ideas.pdf`
- `input/Dinner Ideas - Mains_1.pdf`
- `input/input.json`

---

### Step 2: Build the Docker Image

Open a terminal in the root project directory and run:
```docker build --platform linux/amd64 -t
mysolutionname:somerandomidentifier```

Run the solution using the run command 
```docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --
network none mysolutionname:somerandomidentifier```
