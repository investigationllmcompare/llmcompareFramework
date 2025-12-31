# 🧠 LLM Benchmarking Framework

A Python framework for **benchmarking multiple Large Language Models (LLMs)** across various NLP datasets and tasks — including MMLU, SQuAD, TruthfulQA, RACE, CNN/DailyMail, and more.  
Supports **OpenAI**, **Ollama**, and **Anthropic** model providers.

---

## 🚀 Features

- 📊 Evaluate multiple models across standard NLP benchmarks  
- ⚙️ Unified interface for OpenAI, Ollama, and Anthropic APIs  
- 🧮 Built-in metrics: F1, ROUGE, BLEU, Edit Distance  
- 🗂️ Dataset support via Hugging Face Datasets  
- 📄 Optional PDF summarization for research papers  
- 💾 Automatic CSV export of benchmarking results  

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-benchmarking.git
cd llm-benchmarking

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # (or venv\Scripts\activate on Windows)

# Install dependencies
pip install -r requirements.txt

---

🔑 Setup API Keys

Before running, add your API keys inside the script (or export them as environment variables):

export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

---

🧪 Usage Example

Run the benchmarking script from the command line:

python benchmark.py \
  --models "gpt-4o:openai" "llama3:ollama" \
  --datasets "mmlu" "squad" \
  --output results.csv

---

📊 Output

After completion, results are saved in a CSV file (default: benchmark_results.csv) with the following structure:

dataset	metric	model1	model2	...
squad	f1	0.87	0.83	
mmlu	accuracy	0.74	0.68

---

📈 Extending

To add a new dataset:

Define a new function evaluate_<dataset>() that loads and evaluates it.

Add it to the DATASET_FUNCTIONS dictionary in benchmark.py.

Run the script with --datasets <your_dataset_name>.
