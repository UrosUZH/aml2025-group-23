"""
Text Perturbation and Evaluation Utilities

This module provides tools to simulate perturbations (noise) on text sentences and to evaluate the impact
of these perturbations against reference sentences using standard natural language generation metrics.

---

## Overview

The script supports two main perturbation strategies:

1. **Shuffling inner tokens**: Randomly shuffles tokens inside sentences while optionally keeping the edge words fixed.
2. **Replacing tokens randomly**: Replaces tokens in the sentence with other tokens sampled from the whole corpus.

After perturbing sentences, the script evaluates the degraded outputs against references using:

- BLEU score
- ROUGE-1 and ROUGE-L
- BERTScore (F1)
- Sentence-level cosine similarity using Sentence Transformers

It also includes utilities for saving sentences, generating test data, and exporting evaluation results to CSV.

---

## Example Usage

```bash
python3 Evaluation.py
```
"""


import os
import random
from typing import List, Tuple
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util
import csv
from transformers import logging

# Suppress warnings from transformers library
logging.set_verbosity_error()


def save_lines_to_file(lines: List[str], output_path: str, verbose: bool = True) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line.strip() + '\n')
    if verbose:
        print(f"Saved: {output_path}")



def shuffle_inner_tokens(sentence: str, keep_edges: bool = True, permute_ratio: float = 1.0) -> str:
    """
    Shuffle the inner tokens of a sentence.

    Args:
        sentence (str): Input sentence.
        keep_edges (bool): Whether to keep the first and last tokens fixed. Defaults to True.
        permute_ratio (float): Ratio of inner tokens to shuffle (0 to 1). Defaults to 1.0.

    Returns:
        str: Sentence with shuffled tokens.
    """
    tokens = sentence.strip().split()
    if len(tokens) <= 1 or permute_ratio <= 0:
        return sentence

    indices = list(range(1, len(tokens) - 1)) if keep_edges and len(tokens) > 2 else list(range(len(tokens)))
    count = max(1, int(len(indices) * permute_ratio))

    selected = random.sample(indices, count)
    subset = [tokens[i] for i in selected]
    random.shuffle(subset)

    for idx, token_idx in enumerate(selected):
        tokens[token_idx] = subset[idx]
    return ' '.join(tokens)


def batch_shuffle_inner_tokens(sentences: List[str], keep_edges: bool = True, permute_ratio: float = 1.0) -> List[str]:
    """
    Apply inner token shuffling to a batch of sentences.

    Args:
        sentences (List[str]): List of sentences to shuffle.
        keep_edges (bool): Whether to keep edge tokens fixed. Defaults to True.
        permute_ratio (float): Ratio of tokens to shuffle. Defaults to 1.0.

    Returns:
        List[str]: Shuffled sentences.
    """
    return [shuffle_inner_tokens(s, keep_edges, permute_ratio) for s in sentences]


def replace_with_random_tokens(sentences: List[str], rand_replace_prob: float = 0.3) -> List[str]:
    """
    Replace tokens in each sentence with random tokens from the full token pool.

    Args:
        sentences (List[str]): List of sentences to perturb.
        rand_replace_prob (float): Probability of replacing each token. Defaults to 0.3.

    Returns:
        List[str]: Sentences with randomly replaced tokens.
    """
    all_tokens = [tok for s in sentences for tok in s.strip().split()]
    pool = all_tokens[:]
    randomized_sentences, idx = [], 0

    for sent in sentences:
        tokens = sent.strip().split()
        new_tokens = pool[idx:idx + len(tokens)]
        idx += len(tokens)

        for i in range(len(new_tokens)):
            if random.random() < rand_replace_prob:
                new_tokens[i] = random.choice(all_tokens)
        randomized_sentences.append(' '.join(new_tokens))
    return randomized_sentences

#  Metrics

def compute_bleu_score(ref_path: str, pred_path: str) -> float:
    """
    Compute BLEU score between reference and predicted sentences stored in files.

    Args:
        ref_path (str): Path to the reference text file.
        pred_path (str): Path to the predicted text file.

    Returns:
        float: BLEU score.
    """
    with open(ref_path, 'r', encoding='utf-8') as rf, open(pred_path, 'r', encoding='utf-8') as pf:
        references = [[line.strip().split()] for line in rf]
        predictions = [line.strip().split() for line in pf]
    return corpus_bleu(references, predictions, smoothing_function=SmoothingFunction().method4)


def compute_rouge_scores(ref_path: str, pred_path: str) -> Tuple[float, float]:
    """
    Compute ROUGE-1 and ROUGE-L F1 scores between reference and predicted files.

    Args:
        ref_path (str): Reference file path.
        pred_path (str): Prediction file path.

    Returns:
        Tuple[float, float]: ROUGE-1 F1 and ROUGE-L F1 scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    with open(ref_path, 'r', encoding='utf-8') as rf, open(pred_path, 'r', encoding='utf-8') as pf:
        refs, preds = rf.readlines(), pf.readlines()

    r1_list, rL_list = [], []
    for r, p in zip(refs, preds):
        sc = scorer.score(r.strip(), p.strip())
        r1_list.append(sc['rouge1'].fmeasure)
        rL_list.append(sc['rougeL'].fmeasure)

    return sum(r1_list) / len(r1_list), sum(rL_list) / len(rL_list)


def compute_bertscore_f1(ref_path: str, pred_path: str) -> float:
    """
    Compute BERTScore F1 score between reference and predicted files.

    Args:
        ref_path (str): Reference file path.
        pred_path (str): Prediction file path.

    Returns:
        float: Average BERTScore F1 score.
    """
    with open(ref_path, 'r', encoding='utf-8') as rf, open(pred_path, 'r', encoding='utf-8') as pf:
        refs = [line.strip() for line in rf]
        preds = [line.strip() for line in pf]

    _, _, f1_scores = bert_score(preds, refs, lang="en", rescale_with_baseline=True,
                                 verbose=False)
    return float(f1_scores.mean())

def get_s2v_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load a SentenceTransformer model for cosine similarity computations.

    Args:
        model_name (str): Name of the pretrained model. Defaults to "all-MiniLM-L6-v2".

    Returns:
        SentenceTransformer: Loaded model.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")

def compute_avg_cosine_sim(ref_path: str, pred_path: str) -> float:
    """
    Compute average diagonal cosine similarity between reference and predicted sentences.

    Args:
        ref_path (str): Reference file path.
        pred_path (str): Prediction file path.

    Returns:
        float: Average cosine similarity score.
    """
    s2v_model = get_s2v_model()
  
    with open(ref_path, 'r', encoding='utf-8') as rf, open(pred_path, 'r', encoding='utf-8') as pf:
        refs = [line.strip() for line in rf]
        preds = [line.strip() for line in pf]

    emb_ref = s2v_model.encode(refs, convert_to_tensor=True, show_progress_bar=False)
    emb_pred = s2v_model.encode(preds, convert_to_tensor=True, show_progress_bar=False)
    sim_matrix = util.cos_sim(emb_pred, emb_ref)
    return float(sim_matrix.diag().mean())

def make_sentences():
    """
    Create example reference sentences, generate shuffled and randomized versions,
    and save them to disk.

    Returns:
        Tuple[str, List[str], str]: Paths to shuffled file, list of randomized files, and reference file.
    """
    reference_sentences = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning is fascinating and powerful",
        "openai creates amazing technology that changes the world",
        "natural language processing enables computers to understand text",
        "deep learning models require large amounts of data",
        "artificial intelligence is transforming industries rapidly"
    ]

    output_dir = "mock_sentences"
    os.makedirs(output_dir, exist_ok=True)

    ref_path = os.path.join(output_dir, "reference.txt")
    pred_path = os.path.join(output_dir, "shuffled.txt")

    # Save reference and shuffled
    save_lines_to_file(reference_sentences, ref_path)
    shuffled_refs = batch_shuffle_inner_tokens(reference_sentences, permute_ratio=0.3)
    save_lines_to_file(shuffled_refs, pred_path)

    # Random replacement configurations
    rand_replacement_configs = [(0.3, "rand_03.txt"), (0.5, "rand_05.txt"), (0.9, "rand_09.txt")]
    randomized_paths = []

    for prob, filename in rand_replacement_configs:
        rand_sent = replace_with_random_tokens(reference_sentences, rand_replace_prob=prob)
        path = os.path.join(output_dir, filename)
        save_lines_to_file(rand_sent, path)
        randomized_paths.append(path)
    return pred_path, randomized_paths, ref_path

def run_evaluations():
    """
    Run evaluations on mock sentences by generating shuffled and randomized versions,
    and computing various metrics against the reference sentences.
    """
    pred_path, randomized_paths, ref_path = make_sentences()
    # Gather all eval paths & labels
    all_eval_paths = [pred_path] + randomized_paths + [ref_path]
    eval_labels = ["Shuffled", "Random 30%", "Random 50%", "Random 90%", "Reference"]

    # Print results
    print("\n📊 Evaluation Results:")
    header = f"{"Variant":<15} | {"BLEU":>6} | {"ROUGE-1":>7} | {"ROUGE-L":>7} | {"Sim":>6} | {"BERT-F1":>6}"
    print(header)
    print("-" * len(header))

    for label, path in zip(eval_labels, all_eval_paths):
        bleu = compute_bleu_score(ref_path, path)
        r1, rL = compute_rouge_scores(ref_path, path)
        sim = compute_avg_cosine_sim(ref_path, path)
        bert_f1 = compute_bertscore_f1(ref_path, path)
        
        print(f"{label:<15} | {bleu*100:6.2f} | {r1*100:7.2f} | {rL*100:7.2f} | {sim*100:6.2f} | {bert_f1*100:6.2f}")

def evaluate(
    reference_path: str,
    hypothesis_paths: List[str],
    labels: List[str],
    use_bert: bool = False,
    use_cosine: bool = True,
):
    """
    Evaluate multiple hypothesis files against a reference using various metrics.
    These metrics include BLEU, ROUGE-1, ROUGE-L, BERTScore, and cosine similarity.

    Args:
        reference_path (str): Path to reference file.
        hypothesis_paths (List[str]): Paths to hypothesis files.
        labels (List[str]): Labels for each hypothesis.
        use_bert (bool): Whether to compute BERTScore. Defaults to False.
        use_cosine (bool): Whether to compute cosine similarity. Defaults to True.

    Returns:
        Tuple[List[List[str]], List[str]]: Evaluation results and CSV header.
    """
    results = []

    print("\nEvaluation Results:")
    header = f"{'Variant':<15} | {'BLEU':>6} | {'ROUGE-1':>7} | {'ROUGE-L':>7}"
    csv_header = ["Variant", "BLEU", "ROUGE-1", "ROUGE-L"]
    
    if use_cosine:
        header += f" | {'Cosine':>6}"
        csv_header.append("Cosine")

    if use_bert: 
        header += " | {:>9}".format("BERT-F1")
        csv_header.append("BERT-F1")
    
    print(header)
    print("-" * len(header))

    for label, path in zip(labels, hypothesis_paths):
        bleu = compute_bleu_score(reference_path, path)
        r1, rL = compute_rouge_scores(reference_path, path)

        row = [label, bleu * 100, r1 * 100, rL * 100]
        line = f"{label:<15} | {bleu*100:6.2f} | {r1*100:7.2f} | {rL*100:7.2f}"

        if use_cosine:
            cosine_sim = compute_avg_cosine_sim(reference_path, path)
            line += f" | {cosine_sim*100:6.2f}"
            row.append(cosine_sim * 100)

        if use_bert:
            bert_f1 = compute_bertscore_f1(reference_path, path)
            line += f" | {bert_f1*100:9.2f}"
            row.append(bert_f1 * 100)

        print(line)
        results.append(row)
    return results, csv_header
    

def save_to_csv(data: List[List[str]], header: List[str], output_path: str = "mock_sentences/evaluation_results.csv") -> None:
    """
    Save evaluation results to a CSV file.

    Args:
        data (List[List[str]]): Evaluation results.
        header (List[str]): CSV header.
        output_path (str): Path to output CSV file. Defaults to "mock_sentences/evaluation_results.csv".
    """
    with open(output_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    print(f"Saved results to {output_path}")

if __name__ == '__main__':
    make_sentences()
    results, header = evaluate(
    reference_path="mock_sentences/reference.txt",
    hypothesis_paths=[
        "mock_sentences/reference.txt",
        "mock_sentences/shuffled.txt",
        "mock_sentences/rand_03.txt",
        "mock_sentences/rand_05.txt",
        "mock_sentences/rand_09.txt",
       
    ],
    labels=["Reference", "Shuffled", "Noisy 30%", "Noisy 50%", "Noisy 90%"],
    use_bert=False,
    use_cosine=False
)
    save_to_csv(results, header, output_path="mock_sentences/evaluation_results.csv")
  
    
