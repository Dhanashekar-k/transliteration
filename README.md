# Multilingual Transliteration

## Overview

This project implements a **Transformer-based transliteration model** designed to convert text from **English into multiple target languages**. The model leverages the **Aksharantar dataset**, enabling accurate transliteration across diverse language scripts.

## Dataset Description

The **Aksharantar dataset** serves as a comprehensive benchmark for transliteration, consisting of parallel text pairs from **English to various target languages**. It is widely used for training and evaluating transliteration models.

- **Languages Included:** Hindi, Bengali, Tamil, Telugu, Kannada, Gujarati, Marathi, Malayalam, Punjabi, and more.
- **Data Type:** Parallel transliteration pairs (English → target scripts)
- **Purpose:** Benchmarking transliteration accuracy and training high-quality models.

## Model Used

### **Transformer-Based Model**
- **Encoder-Decoder Architecture:** Efficiently learns language context for transliteration.
- **Self-Attention Mechanism:** Captures dependencies across input sequences.
- **Multi-Head Attention:** Improves model performance on transliteration tasks.

## Usage

1. **Preprocessing:**  
   - Run `data_preprocessing.ipynb` to preprocess the **Aksharantar dataset**.

2. **Training:**  
   - Use `training.py` to train the Transformer model.

3. **Inference:**  
   - Use `inference.py` (located in the `scripts/` folder) to test transliterations on custom inputs.

4. **Logs:**  
   - Training and inference logs are stored in the `logs/` directory.

## Dataset Access

The **Aksharantar dataset** is publicly available at:  
🔗 [Aksharantar Dataset on Hugging Face](https://huggingface.co/datasets/ai4bharat/Aksharantar/tree/main)

---
