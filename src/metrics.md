# Evaluation Metrics for Sentence-Level Generation

In evaluating the quality of generated sentences, particularly in the context of translating sign language videos into natural language, it is important to consider both surface-level overlap and semantic similarity. This document outlines several commonly used metrics, detailing how they function, their advantages and limitations, and their relevance to this task.


https://www.geeksforgeeks.org/nlp/understanding-bleu-and-rouge-score-for-nlp-evaluation/


## BLEU https://www.nltk.org/_modules/nltk/translate/bleu_score.html
 
BLEU (Bilingual Evaluation Understudy) is a precision-based metric commonly used in machine translation. It evaluates how many n-grams in the generated output appear in the reference sentence.

BLEU calculates n-gram precision and applies a brevity penalty if the output is shorter than the reference. It outputs a score between 0 and 1, where higher scores indicate greater overlap with the reference.
 
BLEU is widely used as a benchmark metric in natural language generation tasks and allows for standardized comparisons across systems.

**Pros:**
- Fast to compute and widely supported.
- Standardized and reproducible.
- Effective at capturing exact matches in structured tasks like translation.

**Cons:**
- Penalizes semantically correct paraphrases.
- Ignores synonymy, grammar, and sentence meaning.
- Poor correlation with human judgment on more flexible generation tasks.

**Conclusion:**  
BLEU is useful for benchmarking and comparing systems but should not be relied on in isolation for tasks requiring semantic understanding, such as translating sign language into natural language.

## ROUGE-1 https://www.geeksforgeeks.org/nlp/understanding-bleu-and-rouge-score-for-nlp-evaluation/

ROUGE-1 is a recall-based metric that measures the proportion of unigrams (single words) in the reference that are also present in the generated output.
It compares token-level overlap without considering order or context. It is frequently used in summarization and content coverage evaluations.

ROUGE-1 provides an estimate of how much key vocabulary from the reference is recovered in the generated text.

**Pros:**
- Simple and intuitive to interpret.
- Useful for evaluating content coverage.
- Widely adopted in summarization tasks.

**Cons:**
- Does not consider word order or grammar.
- Ignores synonyms and paraphrases.
- May overestimate quality if key words are present but poorly structured.

**Conclusion:**  
ROUGE-1 is useful for assessing whether generated outputs capture core vocabulary but is insufficient for evaluating semantic quality or fluency.

## ROUGE-L

ROUGE-L measures the longest common subsequence (LCS) between the predicted and reference sentences. It captures both content and sequence alignment. 
By identifying the longest sequence of words appearing in the same order, ROUGE-L accounts for sentence structure more effectively than unigram-based metrics.
Often used in summarization and natural language generation tasks to assess fluency and coherence.

**Pros:**
- Captures in-order phrase and sequence matches.
- Better reflects fluency than unigram-based metrics.
- Simple to compute and interpret.

**Cons:**
- Still insensitive to synonymy and meaning.
- Penalizes valid paraphrasing.
- Surface-level; not suitable for meaning-based evaluation.

**Conclusion:**  
ROUGE-L complements other metrics by evaluating fluency and structure but should be combined with semantic metrics for tasks involving diverse or paraphrased outputs.

## Cosine Similarity (Sentence Embeddings) https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

Cosine similarity measures the angle between vector representations (embeddings) of sentences. It is a semantic metric that captures meaning rather than surface form.
Sentences are encoded into high-dimensional vectors using transformer-based models such as `all-MiniLM-L6-v2`. The cosine of the angle between two vectors indicates their semantic similarity.
Used to assess semantic equivalence between sentences, especially when word choice and structure differ but the meaning is preserved.

**Pros:**
- Captures deep semantic similarity.
- Robust to rephrasing and lexical variation.
- Fast and scalable to large datasets.

**Cons:**
- Less interpretable than token-level metrics.
- Can be fooled by fluent but factually incorrect sentences.
- Quality depends on the embedding model used.

**Conclusion:**  
Cosine similarity with pretrained embeddings is a strong semantic baseline for tasks like sign language translation, where surface forms may vary but meaning must be preserved.

## BERTScore (F1) https://huggingface.co/spaces/evaluate-metric/bertscore

BERTScore evaluates semantic similarity by aligning contextualized token embeddings from models like BERT, RoBERTa, or DeBERTa. It provides precision, recall, and F1 scores.
Each token in the generated sentence is matched to the most similar token in the reference sentence using contextual embeddings. The F1 score summarizes the harmonic mean of precision and recall over these matches. 
Frequently used in machine translation, summarization, and captioning, where semantic fidelity is critical.

**Pros:**
- High correlation with human semantic judgments.
- Sensitive to context, meaning, and token relationships.
- Does not rely on surface word matches.

**Cons:**
- Computationally expensive.
- May require GPU acceleration for efficiency.
- Can be sensitive to minor model variations or versioning.

**Conclusion:**  
BERTScore is a strong semantic evaluator and particularly valuable for final evaluations. However, due to its computational cost, it is best used selectively rather than for high-throughput evaluation.
