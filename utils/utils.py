import numpy as np
import csv

import json
import pandas as pd
import os
from pathlib import Path
import numpy as np
import torch
import numpy as np
import pandas as pd
from pose_format import Pose
import sys
sys.path.insert(1, '/home/signclip/fairseq/examples/MMPT')
os.chdir('/home/signclip/fairseq/examples/MMPT')

# If your functions are in another file, adjust the import accordingly
# from your_module import embed_pose, preprocess_pose
import pickle
import av
import json
import pandas as pd
import unicodedata

from src.Transformer import TransformerDecoder

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def load_top_canditates(output_csv_path, k=10):

    df = pd.read_csv(output_csv_path)
    all_candidate_lists = []

    for _, row in df.iterrows():
        candidate_list = []
        for i in range(1, k+1): 
            label = str(row[f'label_{i}']).lower()
            score = float(row[f'score_{i}'])
            candidate_list.append((label, score))
        all_candidate_lists.append(candidate_list)
    
    return all_candidate_lists
from scipy.special import softmax

def temperature_scaled_softmax(x, temperature=8.0):
    x = np.array(x) / temperature
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def min_max_normalize(x):
    x = np.array(x)
    min_val, max_val = np.min(x), np.max(x)
    if max_val - min_val == 0:
        return np.ones_like(x) / len(x)  # fallback to uniform distribution
    norm = (x - min_val) / (max_val - min_val)
    return norm / norm.sum()

def load_top_candidates(output_csv_path, k=10):
    df = pd.read_csv(output_csv_path)
    all_candidate_lists = []

    for _, row in df.iterrows():
        labels = [str(row[f'label_{i}']).lower() for i in range(1, k + 1)]
        scores = np.array([float(row[f'score_{i}']) for i in range(1, k + 1)])
        probs = temperature_scaled_softmax(scores)
        print(sum(probs))
        candidate_list = list(zip(labels, probs))
        print(candidate_list)
        all_candidate_lists.append(candidate_list)

    return all_candidate_lists, df
import pandas as pd
def retrieve_sentence_map(alignment_csv_path):
    
    columns = [
        "VIDEO_ID",
        "VIDEO_NAME",
        "SENTENCE_ID",
        "SENTENCE_NAME",
        "START",
        "END",
        "SENTENCE"
    ]
    align_df = pd.read_csv(
        alignment_csv_path,
        sep="\t",
        names=columns,
        quoting=3,
        encoding="utf-8",
        on_bad_lines="skip", header=0
    )
    return dict(zip(align_df['SENTENCE_NAME'], align_df['SENTENCE']))


def create_sentence_comparison_csv(
    output_csv_path,
    sentence_name,
    sentence,
    alignment_csv_path,
    save_path,
    beam_size=3,
    k=10,
):
    # Load the sentence map
    sentence_map = retrieve_sentence_map(alignment_csv_path)
    ground_truth = sentence_map.get(sentence_name, "[NOT FOUND]")
    decoded_sentence = sentence if sentence else "[EMPTY]"

    # Prepare the new row
    new_row = {
        "output_csv_path": output_csv_path.name,
        "SENTENCE_NAME": sentence_name,
        "SENTENCE": ground_truth,
        "sentence_decoder": decoded_sentence,
        "beam_size": beam_size,
        "top_k": k, 
    }

    # Try to load existing CSV
    try:
        df = pd.read_csv(save_path)
    except FileNotFoundError:
        df = pd.DataFrame()

    # Check for exact row duplicate
    if not df.empty:
        # Ensure same column order
        df = df[list(new_row.keys())]
        # Convert all rows to dicts and check if the new row is in them
        row_exists = any(df_row == new_row for df_row in df.to_dict(orient="records"))
        if row_exists:
            print("⚠️ Exact entry already exists. Skipping.")
            return

    # Append the new row
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Appended new entry to {save_path} (total: {len(df)} rows)")

    # Backup logic
    if len(df) % 10 == 0:
        backup_path = save_path.with_name(f"{save_path.stem}_{len(df)}{save_path.suffix}")
        df.to_csv(backup_path, index=False)
        print(f"🧷 Backup saved to {backup_path}")


import pandas as pd
from pathlib import Path
from typing import Union
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def simple_tokenize(text: str) -> list:
    return text.lower().split()


def compute_sentence_bleu(reference: str, hypothesis: str) -> float:
    if hypothesis.startswith("hello can"):
            print(hypothesis)
            print(reference)
            
    ref_tokens = [simple_tokenize(reference)]
    ref_tokens2 = simple_tokenize(reference)
    hyp_tokens = simple_tokenize(hypothesis)
    if hypothesis.startswith("hello can"):
            print(hypothesis)
            print(reference)
            print(sentence_bleu(ref_tokens, hyp_tokens))
            print(sentence_bleu(ref_tokens2, hyp_tokens))
          
    
    return sentence_bleu(ref_tokens, hyp_tokens)


def compute_rouge_individual(reference: str, hypothesis: str) -> tuple:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rouge1'].fmeasure, scores['rougeL'].fmeasure


def compute_cosine_similarity(reference: str, hypothesis: str) -> float:
    vec = TfidfVectorizer().fit_transform([reference, hypothesis])
    return cosine_similarity(vec[0:1], vec[1:2])[0, 0]


def evaluate_sentence_csv_rows(csv_path: Union[str, Path], output_path: Union[str, Path]):
    df = pd.read_csv(csv_path)
    df = pd.read_csv(
    csv_path,
    quoting=csv.QUOTE_MINIMAL,
    quotechar='"',
    escapechar='\\',
    dtype=str,
    keep_default_na=False,
    engine='python'  # THIS is critical for newline handling
)

    bleu_scores = []
    rouge1_scores = []
    rougeL_scores = []
    cosine_scores = []

    for _, row in df.iterrows():
        # Use fallback if needed
        reference = row["SENTENCE"]
        hypothesis = row["sentence_decoder"]
        
        if pd.isna(reference) or reference == "[NOT FOUND]":
            reference = " ".join(row["SENTENCE_NAME"].split("_"))
            print(reference)
            

        hypothesis = row["sentence_decoder"]
        
        

        # Compute scores
        bleu = compute_sentence_bleu(reference, hypothesis)
        r1, rL = compute_rouge_individual(reference, hypothesis)
        cosine = compute_cosine_similarity(reference, hypothesis)

        bleu_scores.append(round(bleu * 100, 2))
        rouge1_scores.append(round(r1 * 100, 2))
        rougeL_scores.append(round(rL * 100, 2))
        cosine_scores.append(round(cosine * 100, 2))

    # Append new columns
    df["BLEU"] = bleu_scores
    df["ROUGE-1"] = rouge1_scores
    df["ROUGE-L"] = rougeL_scores
    df["Cosine"] = cosine_scores

    # Save to new CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Row-level evaluation saved to {output_path}")


import pandas as pd
from pathlib import Path
from typing import List
from src.Evaluation import evaluate
def evaluate_and_save(
    save_path: Path,
    output_path: Path = Path("aml/results/evaluation.csv"),
    use_bert: bool = False,
    use_cosine: bool = True,
):
    # Load sentence comparison CSV
    df = pd.read_csv(save_path)

    # Replace missing references with SENTENCE_NAME fallback
    references = df["SENTENCE"].fillna("").replace("[NOT FOUND]", pd.NA)
    references = references.combine_first(df["SENTENCE_NAME"])

    # Use decoder outputs as hypotheses
    hypotheses = df["sentence_decoder"]

    # Save to temp files for evaluation
    ref_tmp = Path("ref.tmp.txt")
    hyp_tmp = Path("hyp.tmp.txt")

    with ref_tmp.open("w", encoding="utf-8") as ref_f, hyp_tmp.open("w", encoding="utf-8") as hyp_f:
        for ref, hyp in zip(references, hypotheses):
            ref_f.write(str(ref).strip() + "\n")
            hyp_f.write(str(hyp).strip() + "\n")

    # Run evaluation
    results, headers = evaluate(
        reference_path=str(ref_tmp),
        hypothesis_paths=[str(hyp_tmp)],
        labels=["Model"],
        use_bert=use_bert,
        use_cosine=use_cosine
    )

    # Remove temp files
    ref_tmp.unlink()
    hyp_tmp.unlink()

    # Append scores to original DataFrame
    scores = results[0][1:]  # Skip "Model" label
    for key, value in zip(headers[1:], scores):
        df[key] = value

    # Save final evaluated CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Evaluation results saved to {output_path}")
import re
def augment_with_window_stride(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    # Extract Wxx and Sxx using regex from output_csv_path column
    def extract_w_s(file_name):
        match = re.search(r'W(\d+)_S(\d+)', file_name)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    # Apply extraction for each row
    df['window_size'], df['stride'] = zip(*df['output_csv_path'].map(extract_w_s))

    # Save to new file
    out_path = Path(output_csv_path)
    df.to_csv(out_path, index=False)
    print(f"✅ Saved with window/stride to {out_path}")
    

def should_skip_decoding(csv_path, output_csv_path, sentence_name, beam_size, top_k):
    if not Path(csv_path).exists():
        return False  # Nothing to skip, file doesn't exist yet

    df = pd.read_csv(csv_path)

    # Check if a row exists with matching criteria
    match = (
        (df["output_csv_path"] == output_csv_path.name)
        & (df["SENTENCE_NAME"] == sentence_name)
        & (df["beam_size"] == beam_size)
        & (df["top_k"] == top_k)
    )

    return match.any()
    
import pandas as pd
def table_summary():

    # Load your evaluated CSV file
    csv_path = "aml/results/sentences_compare_evaluated.csv"
    df = pd.read_csv(csv_path)

    # Group by (window_size, stride, beam_size)
    group_cols = ["window_size", "stride", "beam_size", "top_k"]

    # Select metric columns that exist
    score_columns = [col for col in ["BLEU", "ROUGE-1", "ROUGE-L", "Cosine", "BERT-F1"] if col in df.columns]

    # Group and aggregate
    summary = (
        df.groupby(group_cols)[score_columns]
        .mean()
        .round(2)
        .reset_index()
    )

    # Add count column
    counts = df.groupby(group_cols).size().reset_index(name="count")

    # Merge counts into summary
    summary = pd.merge(summary, counts, on=group_cols)

    # Sort for readability
    summary = summary.sort_values(by=group_cols).reset_index(drop=True)

    # Show the summary table
    print(summary.to_string(index=False))