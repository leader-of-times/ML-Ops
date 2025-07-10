# 📝 Grammar Scoring Engine

A robust grammar scoring and correction engine that processes audio inputs, transcribes them, corrects grammatical errors, and evaluates the overall language quality using weighted linguistic metrics.

---

## 🚀 Overview

This project provides an end-to-end pipeline to:

- Convert audio input to text using automatic speech recognition (ASR)
- Correct grammatical errors in the transcription using a transformer-based model
- Evaluate the transcription based on:
  - BERTScore similarity
  - Word error rate
  - Sentence structure analysis
  - Grammatical complexity
- Generate a composite grammar score on a scale of 0 to 10

The system is designed to be used in applications like automated language evaluation, communication training, and educational feedback tools.

---

## 🔍 Key Features

- **Speech-to-Text:** Uses OpenAI's `whisper-small` model for ASR.
- **Grammar Correction:** Uses the `vennify/t5-base-grammar-correction` model via Happy Transformer.
- **Language Quality Scoring:** Combines multiple linguistic metrics with adjustable weights.
- **Error Penalty Adjustment:** Penalizes scores based on error rate and similarity thresholds.
- **Modular Pipeline:** Easily extensible and customizable.

---

## ⚙️ Tech Stack

- **Python**
- **spaCy** for NLP parsing
- **Happy Transformer** for grammar correction
- **Transformers** (Hugging Face) for ASR
- **BERTScore** for text similarity evaluation
- **JiWER** for word error rate calculation
- **NumPy**, **JSON**, and other standard libraries

---

## 🧠 Model Deployment

The model is deployed using **Modelbit**, allowing for fast and scalable inference via an API or SDK.

---

## 🏁 Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- `torch`, `transformers`, `happytransformer`, `spacy`, `jiwer`, `bert_score`, `numpy`

Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

## 📦 Usage

You can run inference by passing base64-encoded or JSON-encoded audio data (as an array of float32 samples at 16kHz). The pipeline will handle transcription, correction, and scoring.

---

## 📊 Output

The engine returns a JSON object containing:

- Original transcription  
- Corrected text  
- Individual metric scores  
- Final grammar score (0–10)  
- Timestamp  

---

## 🧪 Testing & Validation

To test the engine, pass in sample audio arrays and verify:

- Accurate transcription  
- Logical and grammatically correct corrections  
- Reasonable grammar scores based on intentional input errors  

---

## 🛠 Troubleshooting

- Ensure the required models (`en_core_web_sm`, `whisper-small`, and T5) are downloaded.  
- If using a GPU, make sure `torch.cuda` is available and correctly configured.  
- Check for missing dependencies in your environment.  

---

## 🧾 License

This project is open-source and available under the MIT License.
