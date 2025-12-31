import json
import pandas as pd
from datasets import load_dataset, Dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import ollama
import re
import Levenshtein
import time
import openai
import argparse
import os
import fitz
import random
from collections import defaultdict, Counter
import anthropic
import string

# ---------- Core Model Interface ----------
def get_model_answer(prompt, model_name, provider, system_prompt=None):
    if provider == "ollama":
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = ollama.chat(
            model=model_name,
            messages=messages,
            options={"temperature": 0}  
        )
        return response['message']['content'].strip()
    
    elif provider == "openai":
        openai.api_key = ""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
    
        response = openai.chat.completions.create(
            model=model_name,  
            messages=messages,
            temperature=0.0
        )
    
        return response.choices[0].message.content.strip()

    elif provider == "anthropic":
        client = anthropic.Anthropic(api_key="")

        response = client.messages.create(
            model=model_name,
            temperature=0,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text.strip()


# ---------- Evaluation Metrics ----------
def normalize_answer(s):
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

def compute_f1(prediction, ground_truth):
    """Compute the token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    common = set(pred_tokens) & set(truth_tokens)
    num_same = sum(min(pred_tokens.count(w), truth_tokens.count(w)) for w in common)

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)

def compute_rouge(prediction, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores

def compute_bleu(prediction, reference):
    reference_tokens = reference.split()
    prediction_tokens = prediction.split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smoothie)

def compute_edit_distance(prediction, reference):
    return Levenshtein.distance(prediction.lower(), reference.lower())

# ---------- Dataset Evaluations ----------

def models_benchmarking(model_provider_pairs, datasets):
    results = []

    for model_name, provider in model_provider_pairs:
        print(f"\n🔍 Evaluating model: {model_name} (provider: {provider})")
        for dataset_func in datasets:
            dataset_name = dataset_func.__name__
            print(f"📂 Dataset: {dataset_name}")

            start = time.time()
            try:
                raw_metrics = dataset_func(model_name, provider=provider)
                metrics = {}

                # Flatten nested dictionaries
                for key, value in raw_metrics.items():
                    if isinstance(value, dict):
                        for subkey, subval in value.items():
                            metrics[f"{key}_{subkey}"] = subval
                    else:
                        metrics[key] = value

            except Exception as e:
                print(f"❌ Error evaluating {model_name} on {dataset_name}: {e}")
                metrics = {}

            end = time.time()
            elapsed = round(end - start, 2)

            result = {
                "model": model_name,
                "provider": provider,
                "dataset": dataset_name,
                "time_sec": elapsed,
                **metrics
            }
            results.append(result)

    df = pd.DataFrame(results)

    # Identify all metric columns dynamically (except model/provider/dataset/time_sec)
    core_cols = {"model", "provider", "dataset", "time_sec"}
    metric_cols = [col for col in df.columns if col not in core_cols]

    # Convert and round numeric columns
    for col in metric_cols + ["time_sec"]:
        df[col] = pd.to_numeric(df[col], errors='coerce').round(4)

    # Pivot for model comparison
    melted = df.melt(id_vars=["model", "provider", "dataset"],
                     value_vars=metric_cols + ["time_sec"],
                     var_name="metric", value_name="value")

    pivoted = melted.pivot_table(index=["dataset", "metric"],
                                 columns=["model"],
                                 values="value",
                                 aggfunc="first").reset_index()

    print("\n📊 Model Evaluation Comparison:\n")
    return pivoted

def evaluate_mmlu(model_name, provider):
    # Load MMLU test set
    dataset = load_dataset("cais/mmlu", "all", split="test")

    # Subjects to exclude
    excluded_subjects = {
        "college_mathematics",
        "elementary_mathematics",
        "high_school_mathematics",
        "abstract_algebra"
    }

    # Filter out excluded subjects
    non_math_data = [ex for ex in dataset if ex['subject'] not in excluded_subjects]

    # Group examples by subject
    subject_buckets = defaultdict(list)
    for example in non_math_data:
        subject_buckets[example["subject"]].append(example)

    # Balanced sampling based on a total sample limit
    sample_limit = 1000
    included_subjects = list(subject_buckets.keys())
    samples_per_subject = max(1, sample_limit // len(included_subjects))  # Avoid zero

    random.seed(42)
    balanced_samples = []

    for subject in included_subjects:
        examples = subject_buckets[subject]
        if len(examples) > 0:
            samples = examples if len(examples) <= samples_per_subject else random.sample(examples, samples_per_subject)
            balanced_samples.extend(samples)

    # Truncate in case we go slightly over the limit
    balanced_samples = balanced_samples[:sample_limit]
    dataset = Dataset.from_list(balanced_samples)

    # Track accuracy
    total_correct = 0
    total_count = 0
    subject_accuracies = defaultdict(lambda: {"correct": 0, "total": 0})

    for idx, example in enumerate(dataset, start=1):
        subject = example['subject']
        question = example['question']
        choices = example['choices']
        correct_answer = example['answer']

        prompt = f"""You are taking a multiple-choice test. Read the question and choose the correct answer from the given options.

Question:
{question}

Choices:
0. {choices[0]}
1. {choices[1]}
2. {choices[2]}
3. {choices[3]}

Respond with only the number (0, 1, 2, or 3) corresponding to the correct answer. Do not include any explanation."""

        model_answer = get_model_answer(prompt, model_name, provider)
        match = re.search(r'\d+', model_answer)

        is_correct = match and int(match.group()) == correct_answer
        if is_correct:
            total_correct += 1
            subject_accuracies[subject]["correct"] += 1

        subject_accuracies[subject]["total"] += 1
        total_count += 1

        print(f"Q{idx}: Subject = {subject} | Model Answer = {model_answer.strip()} | "
              f"Correct Answer = {correct_answer} | {'✅ Correct' if is_correct else '❌ Incorrect'}")

    # Final accuracy computation
    overall_accuracy = total_correct / total_count
    per_subject_accuracy = {
        subject: acc["correct"] / acc["total"]
        for subject, acc in subject_accuracies.items()
    }

    return {
        "accuracy": overall_accuracy,
        "accuracy_subject": per_subject_accuracy
    }


def evaluate_qmsum(model_name, provider):
    dataset = load_dataset("pszemraj/qmsum-cleaned", split="validation")

    def split_input(example):
        query, transcript = example['input'].split('\n', 1)
        example['query'] = query
        example['transcript'] = transcript
        return example

    dataset = dataset.map(split_input)

    rouge_total = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    total = 0

    for example in dataset.select(range(2)):
        query = example['query']
        transcript = example['transcript']
        reference = example['output']
        prompt = f"""Given the following meeting transcript and a specific question, generate a concise and complete summary that directly answers the question using only relevant details from the meeting.
        
        Meeting Transcript:
        {transcript}
        
        Question:
        {query}
        
        Answer:"""

        model_summary = get_model_answer(prompt, model_name, provider)
        
        print(f"Model Answer: {model_summary}")
        print(f"choice: {reference}")

        scores = compute_rouge(model_summary, reference)

        for key in rouge_total:
            rouge_total[key] += scores[key].fmeasure
        total += 1

    return {
        "rouge1": rouge_total["rouge1"] / total,
        "rouge2": rouge_total["rouge2"] / total,
        "rougeL": rouge_total["rougeL"] / total,
    }

def evaluate_truthfulqa_mc(model_name, provider):
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
    correct_predictions = 0
    total = 0

    for row in dataset:
        question = row["question"]
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        correct_indices = [i for i, l in enumerate(labels) if l == 1]

        prompt = f"""Answer the following multiple-choice question. Select the most truthful and accurate option.

Question: {question}

Choices:
{chr(10).join([f"{i}: {choice}" for i, choice in enumerate(choices)])}

Answer only with the number corresponding to the correct choice (0-{len(choices)-1})."""

        model_answer = get_model_answer(prompt, model_name, provider)

        print(f"Model Answer: {model_answer}")
        print(f"choice: {correct_indices}")
        
        match = re.search(r"\d+", model_answer)
        model_index = int(match.group()) if match else -1

        if model_index in correct_indices:
            correct_predictions += 1
        total += 1

    accuracy = correct_predictions / total
    return {"accuracy": accuracy}

def evaluate_truthfulqa_gen(model_name, provider):
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    examples = list(dataset)
    total = len(examples)
    correct = 0

    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    for example in examples:
        question = example['question']
        best_answer = example['best_answer']
        correct_answers = example['correct_answers']
        incorrect_answers = example['incorrect_answers']

        # Calculate the maximum word count among all reference answers
        reference_answers = [best_answer] + correct_answers + incorrect_answers
        max_word_count = max(len(ref.split()) for ref in reference_answers)

        # Update the prompt to limit model answer length to the maximum word count
        prompt = f"""Question: {question}
        
Provide a concise and accurate answer. No explanations. Your answer should be no longer than {max_word_count} words."""
        
        # Get the model's answer
        model_answer = get_model_answer(prompt, model_name, provider)

        print(f"Model Answer: {model_answer}")
        print(f"Best Answer: {best_answer}")
        print(f"Correct Answers: {correct_answers}")
        print(f"Incorrect Answers: {incorrect_answers}")
        
        def score_refs(hypo, refs):
            scores = [scorer.score(hypo, ref)["rouge1"].fmeasure for ref in refs]
            return max(scores) if scores else 0.0

        # Score the model answer against the best, correct, and incorrect answers
        score_best = score_refs(model_answer, [best_answer])
        score_correct = score_refs(model_answer, correct_answers)
        score_incorrect = score_refs(model_answer, incorrect_answers)

        # Count the answer as correct if it matches the best or correct answers
        if score_best >= score_correct and score_best >= score_incorrect:
            correct += 1
        elif score_correct >= score_incorrect:
            correct += 1

    return {"accuracy": correct / total}

def evaluate_race(model_name, provider):
    # Load full validation set
    dataset = load_dataset("race", "all", split="test")

    # Separate middle and high school questions
    middle_examples = [item for item in dataset if "middle" in item['example_id']]
    high_examples = [item for item in dataset if "high" in item['example_id']]

    # Set total sample limit
    limit_samples = 1000
    base = limit_samples // 2
    extra = limit_samples % 2

    # Randomly assign extra sample to either group
    extra_middle = random.choice([True, False]) if extra else False
    num_middle = base + (1 if extra_middle else 0)
    num_high = base + (0 if extra_middle else 1)

    # Safety check
    if len(middle_examples) < num_middle or len(high_examples) < num_high:
        print("⚠️ Not enough middle or high school samples to meet the required limit.")
        return {"accuracy": None, "samples": 0}

    # Sample from both groups
    random.seed(42)
    sampled_middle = random.sample(middle_examples, num_middle)
    sampled_high = random.sample(high_examples, num_high)
    sampled_data = sampled_middle + sampled_high
    random.shuffle(sampled_data)

    correct = 0
    total = len(sampled_data)
    letter_to_index = {'A': '0', 'B': '1', 'C': '2', 'D': '3'}

    for item in sampled_data:
        article = item['article']
        question = item['question']
        options = item['options']
        answer_letter = item['answer'].strip().upper()
        answer = letter_to_index.get(answer_letter)

        if answer is None:
            print(f"⚠️ Invalid answer format: {item['answer']}")
            continue

        prompt = f"""Read the following passage and answer this multiple choice question. 
Write only the number (0, 1, 2, or 3) of the correct answer, without explaining.

Passage:
{article}

Question:
{question}

Options:
0. {options[0]}
1. {options[1]}
2. {options[2]}
3. {options[3]}

Answer:"""

        raw_response = get_model_answer(prompt, model_name, provider)
        cleaned_response = raw_response.strip()
        match = re.search(r'\b([0-3])\b', cleaned_response)
        selected = match.group(1) if match else None

        print(f"Model Answer: {cleaned_response}")
        print(f"Correct Answer Index: {answer}\n")

        if selected == answer:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    return {
        "accuracy": accuracy
    }


def evaluate_cnn_dailymail(model_name, provider):
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:1000]")  
    rouge1_total = 0.0
    rouge2_total = 0.0
    rougeL_total = 0.0
    total = len(dataset)

    for item in dataset:
        article = item['article']
        reference = item['highlights']
        target_len = len(reference.split())

        prompt = f"""Here is a news article:

{article}

Please write a short summary for the article in approximately {target_len} words.
Ensure the summary is concise, factually accurate, and captures the main points.
Summary:"""

        summary = get_model_answer(prompt, model_name, provider)

        print(f"Model Answer: {summary}")
        print(f"reference: {reference}")

        rouge = compute_rouge(summary, reference)
        edit = compute_edit_distance(summary, reference)

        rouge1_total += rouge["rouge1"].fmeasure
        rouge2_total += rouge["rouge2"].fmeasure
        rougeL_total += rouge["rougeL"].fmeasure

    return {
        "rouge1": rouge1_total / total,
        "rouge2": rouge2_total / total,
        "rougeL": rougeL_total / total,
    }

def evaluate_squad(model_name, provider):
    squad_dataset = load_dataset("squad", split="validation[:1000]")  

    f1_total = 0.0
    exact_match_total = 0
    total = len(squad_dataset)

    for i, item in enumerate(squad_dataset):
        context = item['context']
        question = item['question']
        answers = item['answers']['text']
        
        # Prompt
        prompt = f"""Context:
{context}

Extract from the context the shortest answer possible to the following question. Avoid any unnecessary detail.

Question: {question}

Answer:"""
        
        model_answer = get_model_answer(prompt, model_name, provider).strip()
        norm_model_answer = normalize_answer(model_answer)

        # Metrics
        best_f1 = max(compute_f1(norm_model_answer, ref) for ref in answers)
        is_exact_match = any(norm_model_answer == normalize_answer(ref) for ref in answers)

        f1_total += best_f1
        exact_match_total += int(is_exact_match)

        print(f"[{i+1}/{total}] EM: {is_exact_match}, F1: {best_f1:.3f}")
        print(f"Model: {model_answer}")
        print(f"Refs: {answers}\n")

    accuracy = exact_match_total / total
    avg_f1 = f1_total / total

    return {
        "accuracy": round(accuracy, 4),
        "f1": round(avg_f1, 4)
    }

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text.strip()

def evaluate_pdf_summaries(model, provider, output_path="pdf_summaries.csv"):
    folder_path = r"C:\Users\afons\Desktop\Tese\papers" 
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    results = []

    if not pdf_files:
        print("⚠️ No PDF files found in the folder.")
        return pd.DataFrame()

    # Process each PDF file
    for idx, pdf_path in enumerate(pdf_files, start=1):
        print(f"\n📘 Processing paper {idx}/{len(pdf_files)}: {os.path.basename(pdf_path)}")
        paper_text = extract_text_from_pdf(pdf_path)

        # Construct the prompt for the model
        prompt = f"""Summarize the following scientific paper clearly and accurately.
Focus on the core contributions, methodology, key findings, and significance.
Use precise, technical language, suitable for graduate-level audience.
Keep the summary within 400 words.


Paper:
{paper_text}

Summary:"""

        # Get the model's response (summary)
        summary = get_model_answer(prompt, model, provider)

        # Append results
        results.append({
            "model": model,
            "provider": provider,
            "paper": os.path.basename(pdf_path),
            "summary": summary
        })

    # Convert results to DataFrame
    summary_df = pd.DataFrame(results)

    # Append to CSV if it exists, otherwise create it
    if os.path.exists(output_path):
        summary_df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        summary_df.to_csv(output_path, index=False)

    print(f"\n✅ Summaries saved to {output_path}")
    return summary_df

DATASET_FUNCTIONS = {
    "mmlu": evaluate_mmlu,
    "qmsum": evaluate_qmsum,
    "truthfulqa_mc": evaluate_truthfulqa_mc,
    "truthfulqa_gen": evaluate_truthfulqa_gen,
    "squad": evaluate_squad,
    "race": evaluate_race,
    "cnn_dailymail": evaluate_cnn_dailymail,
    "pdf_summaries": evaluate_pdf_summaries
}

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark multiple LLMs on multiple datasets.")
    parser.add_argument(
        "--models", 
        nargs="+", 
        required=True,
        help="List of models with providers in the format model:provider (e.g., gpt-4o:openai llama3:ollama)"
    )
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        choices=list(DATASET_FUNCTIONS.keys()),
        required=True,
        help="List of datasets to evaluate (e.g., mmlu triviaqa race)"
    )
    parser.add_argument(
        "--output", 
        default="benchmark_results.csv",
        help="Path to save the output CSV"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    model_provider_pairs = []
    for entry in args.models:
        try:
            model, provider = entry.split(":")
            model_provider_pairs.append((model, provider))
        except ValueError:
            raise ValueError(f"Invalid model format: {entry}. Use model:provider")

    selected_datasets = []
    for name in args.datasets:
        if name == "pdf_summaries":
            for model, provider in model_provider_pairs:
                evaluate_pdf_summaries(model, provider)
            print("✅ PDF summarization completed.")
        else:
            selected_datasets.append(DATASET_FUNCTIONS[name])

    if selected_datasets:
        result_table = models_benchmarking(model_provider_pairs, selected_datasets)
        result_table.to_csv(args.output, index=False)
        print(f"\n✅ Results saved to {args.output}")


