import os
import random
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def write_sentences_to_file(sentences, filepath):
    """Write list of sentences to a file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for sent in sentences:
            f.write(sent.strip() + '\n')
    print(f"Sentences saved to {filepath}")

def permute_sentence_words(sentence, keep_edges=True, permute_ratio=1.0):
    words = sentence.strip().split()
    if len(words) <= 1 or permute_ratio <= 0:
        return sentence

    if keep_edges and len(words) > 2:
        middle = words[1:-1]
        num_to_permute = max(1, int(len(middle) * permute_ratio))
        indices = list(range(len(middle)))
        random.shuffle(indices)
        permute_indices = indices[:num_to_permute]

        to_permute = [middle[i] for i in permute_indices]
        random.shuffle(to_permute)

        for idx, pi in enumerate(permute_indices):
            middle[pi] = to_permute[idx]

        permuted_words = [words[0]] + middle + [words[-1]]
    else:
        num_to_permute = max(1, int(len(words) * permute_ratio))
        indices = list(range(len(words)))
        random.shuffle(indices)
        permute_indices = indices[:num_to_permute]

        to_permute = [words[i] for i in permute_indices]
        random.shuffle(to_permute)

        permuted_words = words[:]
        for idx, pi in enumerate(permute_indices):
            permuted_words[pi] = to_permute[idx]

    return ' '.join(permuted_words)

def permute_sentences(sentences, keep_edges=True, permute_ratio=1.0):
    return [permute_sentence_words(sent, keep_edges, permute_ratio) for sent in sentences]

def write_golden_and_permuted_files_modular(golden_sentences, output_golden_path, output_permuted_path, 
                                            keep_edges=True, permute_ratio=0.5):
    write_sentences_to_file(golden_sentences, output_golden_path)

    permuted = permute_sentences(golden_sentences, keep_edges=keep_edges, permute_ratio=permute_ratio)
    write_sentences_to_file(permuted, output_permuted_path)

def noisy_permutation(sentences, replace_prob=0.3):
    all_words = [word for sent in sentences for word in sent.strip().split()]
    permuted_pool = all_words[:]
    random.shuffle(permuted_pool)

    new_sentences = []
    idx = 0
    for sent in sentences:
        length = len(sent.strip().split())
        new_words = permuted_pool[idx:idx+length]
        idx += length
        
        for i in range(len(new_words)):
            if random.random() < replace_prob:
                new_words[i] = random.choice(all_words)
        new_sentences.append(' '.join(new_words))
    return new_sentences

def evaluate_bleu(golden_file, pred_file):
    with open(golden_file, 'r', encoding='utf-8') as f1, open(pred_file, 'r', encoding='utf-8') as f2:
        references = [line.strip().split() for line in f1]
        candidates = [line.strip().split() for line in f2]

    list_of_references = [[ref] for ref in references]
    smoothie = SmoothingFunction().method4
    score = corpus_bleu(list_of_references, candidates, smoothing_function=smoothie)
    return score

def evaluate_rouge(golden_file, pred_file):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    with open(golden_file, 'r', encoding='utf-8') as f1, open(pred_file, 'r', encoding='utf-8') as f2:
        references = f1.readlines()
        candidates = f2.readlines()

    scores = {'rouge1': [], 'rougeL': []}
    for ref, pred in zip(references, candidates):
        score = scorer.score(ref.strip(), pred.strip())
        scores['rouge1'].append(score['rouge1'].fmeasure)
        scores['rougeL'].append(score['rougeL'].fmeasure)

    avg_rouge1 = sum(scores['rouge1']) / len(scores['rouge1'])
    avg_rougeL = sum(scores['rougeL']) / len(scores['rougeL'])
    return avg_rouge1, avg_rougeL


if __name__ == '__main__':
    golden_sents = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning is fascinating and powerful",
        "openai creates amazing technology that changes the world",
        "natural language processing enables computers to understand text",
        "deep learning models require large amounts of data",
        "artificial intelligence is transforming industries rapidly"
    ]

    # Create output folder if not exists
    out_dir = "mock_sentences"
    os.makedirs(out_dir, exist_ok=True)

    golden_path = os.path.join(out_dir, "golden.txt")
    permuted_path = os.path.join(out_dir, "permuted.txt")

    # Write golden and permuted sentences
    write_golden_and_permuted_files_modular(golden_sents, golden_path, permuted_path, permute_ratio=0.3)

    # Generate noisy permuted sentences with different replacement probabilities
    noisy_sents03 = noisy_permutation(golden_sents, replace_prob=0.3)
    noisy_sents05 = noisy_permutation(golden_sents, replace_prob=0.5)
    noisy_sents09 = noisy_permutation(golden_sents, replace_prob=0.9)

    noisy_files = []
    for sents, filename in zip([noisy_sents03, noisy_sents05, noisy_sents09], 
                               ["noisy_permuted_03.txt", "noisy_permuted_05.txt", "noisy_permuted_09.txt"]):
        file_path = os.path.join(out_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            for sent in sents:
                f.write(sent + "\n")
        noisy_files.append(file_path)
        print(f"Noisy permuted sentences saved to {file_path}")

    # Evaluate all files against golden
    results = []
    eval_files = [permuted_path] + noisy_files + [golden_path]
    descs = ["Permuted", "Noisy 30%", "Noisy 50%", "Noisy 90%", "Golden (self)"]

    for desc, fname in zip(descs, eval_files):
        bleu_score = evaluate_bleu(golden_path, fname)
        rouge1, rougeL = evaluate_rouge(golden_path, fname)
        results.append((desc, bleu_score, rouge1, rougeL))

    # Print summary table
    print("\nEvaluation Results:")
    print(f"{'Dataset':<15} | {'BLEU':>6} | {'ROUGE-1':>7} | {'ROUGE-L':>7}")
    print("-" * 45)
    for desc, bleu, r1, rL in results:
        print(f"{desc:<15} | {bleu*100:6.2f} | {r1*100:7.2f} | {rL*100:7.2f}")
