# Fair AI-Powered Resume Matcher

This project uses **BERT embeddings** to match **resumes with job descriptions** based on **semantic similarity** rather than keyword matching.

## 🚀 Features
- Uses **sentence-transformers (MiniLM-BERT)** to generate job and resume embeddings.
- Computes **cosine similarity** between job descriptions and resumes.
- Outputs **best-matching jobs** for each resume.

## 🔧 Installation
```bash
pip install torch sentence-transformers pandas
