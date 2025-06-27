import os
import random
from typing import List, Tuple
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util



def save_lines_to_file(lines: List[str], output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line.strip() + '\n')
    print(f"Saved: {output_path}")



def shuffle_inner_tokens(sentence: str, keep_edges: bool = True, permute_ratio: float = 1.0) -> str:
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
    return [shuffle_inner_tokens(s, keep_edges, permute_ratio) for s in sentences]


def replace_with_random_tokens(sentences: List[str], rand_replace_prob: float = 0.3) -> List[str]:
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
    with open(ref_path, 'r', encoding='utf-8') as rf, open(pred_path, 'r', encoding='utf-8') as pf:
        references = [[line.strip().split()] for line in rf]
        predictions = [line.strip().split() for line in pf]
    return corpus_bleu(references, predictions, smoothing_function=SmoothingFunction().method4)


def compute_rouge_scores(ref_path: str, pred_path: str) -> Tuple[float, float]:
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
    with open(ref_path, 'r', encoding='utf-8') as rf, open(pred_path, 'r', encoding='utf-8') as pf:
        refs = [line.strip() for line in rf]
        preds = [line.strip() for line in pf]

    _, _, f1_scores = bert_score(preds, refs, lang="en", rescale_with_baseline=True,
                                 verbose=False)
    return float(f1_scores.mean())

# --- Fast Cosine Similarity Setup ---
from transformers import logging
logging.set_verbosity_error()
s2v_model = SentenceTransformer("all-MiniLM-L6-v2")
# s2v_model = SentenceTransformer("all-mpnet-base-v2")

def compute_avg_cosine_sim(ref_path: str, pred_path: str) -> float:
  
    with open(ref_path, 'r', encoding='utf-8') as rf, open(pred_path, 'r', encoding='utf-8') as pf:
        refs = [line.strip() for line in rf]
        preds = [line.strip() for line in pf]

    emb_ref = s2v_model.encode(refs, convert_to_tensor=True, show_progress_bar=False)
    emb_pred = s2v_model.encode(preds, convert_to_tensor=True, show_progress_bar=False)
    sim_matrix = util.cos_sim(emb_pred, emb_ref)
    return float(sim_matrix.diag().mean())

# Main Execution

def make_sentences():
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

    print("\nEvaluation Results:")
    header = f"{'Variant':<15} | {'BLEU':>6} | {'ROUGE-1':>7} | {'ROUGE-L':>7}"
    if use_cosine:
        header = header + f" | {'Cosine':>6}"
    if use_bert: 
        header = header + (" | {:>9}".format("BERT-F1"))
    
    print(header)
    print("-" * len(header))

    for label, path in zip(labels, hypothesis_paths):
        bleu = compute_bleu_score(reference_path, path)
        r1, rL = compute_rouge_scores(reference_path, path)
        
        line = f"{label:<15} | {bleu*100:6.2f} | {r1*100:7.2f} | {rL*100:7.2f}"
        if use_cosine:
            cosine_sim = compute_avg_cosine_sim(reference_path, path)
            line += f" | {cosine_sim*100:6.2f}"
        if use_bert:
            bert_f1 = compute_bertscore_f1(reference_path, path)
            line += f" | {bert_f1*100:9.2f}"

        print(line)



if __name__ == '__main__':
    make_sentences()
    evaluate(
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
