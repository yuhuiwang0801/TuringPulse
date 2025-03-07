# 🤖 LLM_Comparison_Project
🚀 Fine-tuning and Comparing Multiple Language Models (LLMs)

This repository contains scripts and resources for comparing the performance of various language models using different evaluation metrics and visualizations. The goal is to help users understand the strengths and weaknesses of each model based on their requirements.

---

## 📌 Project Overview
This project evaluates multiple LLMs using real-world tasks, focusing on:
- Accuracy and performance across different datasets
- Resource efficiency (memory and processing time)
- Visualization of performance metrics through graphs

The models are fine-tuned using specific datasets and evaluated with consistent parameters to ensure fair comparisons.

---

##🔗 Model Link
You can access the interactive model demo at the link below:
🔗 LLM Comparison Demo

---

## 🔗 Model Comparisons
Performance results and comparisons are visualized in:
- **model_performance_comparison.png**: A graph showing accuracy, speed, and memory usage for each model.

---

## 🔹 Features
- 🏅 **Accuracy Comparison:** Evaluate precision, recall, and F1 scores for each model.
- 🚀 **Resource Efficiency:** Measure memory usage and processing time.
- 📊 **Visualizations:** Generate graphs using Python's plotting libraries.
- 📋 **Multiple Model Support:** Compare OpenAI, AWS Bedrock, and other models.
- 🔄 **Streamlit Integration:** An interactive web interface for exploring results.

---

## 🛠 How to Use

### 1️⃣ Load the fine-tuned model
To use the fine-tuned model, run:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_name = "your-huggingface-username/Your-Model-Name"

tokenizer = AutoTokenizer.from_pretrained(repo_name)
model = AutoModelForCausalLM.from_pretrained(repo_name)

model.eval()
```

---

### 2️⃣ Run inference
```python
prompt = "What are the key differences between various LLMs?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200)

response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Model Response:", response)
```

---

### 3️⃣ Visualize Results
Run the graph generation script to see performance comparisons:
```bash
python LLMgraph.py
```
Check the generated `model_performance_comparison.png` for detailed insights.

---

## 📂 Files in this Repository

| File                              | Description                                                 |
|-----------------------------------|-------------------------------------------------------------|
| **LLM.py**                        | Main script for model evaluation and comparison.             |
| **LLMgraph.py**                   | Script to generate comparison graphs.                        |
| **model_performance_comparison.png** | Visual comparison of model performances.                     |
| **requirements.txt**              | Python dependencies for the project.                         |
| **.devcontainer/**                | Configuration for remote development using VS Code.         |

---

## 📊 Training Details
- **Base Models:** OpenAI, AWS Bedrock, and others
- **Evaluation Metrics:** Accuracy, precision, recall, memory usage, processing time
- **Environment:** Python, Streamlit, and relevant libraries

---

## 📢 Acknowledgments
- **OpenAI:** For API access and documentation.
- **AWS Bedrock:** For providing additional model resources.
- **Streamlit:** For the web interface framework.

