# 🌟 Arabic Sentiment Analysis

A complete sentiment analysis system for Arabic text, supporting both **Arabic script** and **Franco-Arabic** (Arabic written with Latin letters and numbers).

## ✨ Features

- 🇸🇦 **Arabic NLP** - Full preprocessing pipeline for Arabic text
- 🔤 **Franco-Arabic Support** - Handles dialect written with Latin script (e.g., "ana ta3ban")
- 🧠 **Hybrid Embeddings** - Word2Vec + TF-IDF weighted vectors
- 🤖 **Neural Network Classifier** - 3-layer deep learning model
- 🖥️ **Gradio Web UI** - Easy-to-use interface
- 📊 **Visualization** - Training history and confusion matrix

## 🎯 Performance

| Metric | Score |
|--------|-------|
| Architecture | 128 → 64 → 32 → 3 |
| Embeddings | Word2Vec (300d) |
| Classes | Negative, Neutral, Positive |

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
