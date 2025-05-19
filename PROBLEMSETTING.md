## Team members

- **Uros Dimitrijevic**: uros.dimitrijevic@uzh.ch | 15-936-834
- **Roham Zendehdel Nobari**: roham.zendehdelnobari@uzh.ch | 23-755-432
- **Mohammad Mahdi Hejazi**: mohammadmahdi.hejazi@uzh.ch | 24-748-998

# Motivation & Goal

With technology increasingly central to daily life and faster, more responsive tools replacing legacy systems, our focus is on bridging the gap between hearing-impaired individuals and computers to help them relay their thoughts more quickly and effectively.
The primary goal is to explore whether pretrained sign-to-text models, specifically those trained at the gloss level, can generalize to sentence-level generation in a zero-shot setup. This serves as a step toward understanding the limits and capabilities of current pretrained models in bridging pose-based sign language data and spoken language output.

# Problem Setting

We address the task of translating sign language video sequences into natural language sentences. Formally, let  

$F = <f_1, f_2, ...f_N>$

be the frames of our video sequence. Our task is to generate a sequence of gloss 

$G = <g_1, g_2, ..., g_M>$

which forms a meaningful sentence describing what is being said in the video, and each gloss $g_i$ is a word from our vocabulary $V$.
Let $f: ℝ^N → V^M$ denote the model used for this mapping. We investigate whether a pretrained model $f$, initially trained to associate pose data with single-word glosses, can generalize to sentence-level generation in a **zero-shot** setting.

# Approach

We use a pretrained model based on the SignCLIP architecture. While SignCLIP was trained to align isolated signs with textual labels, we aim to evaluate its ability to produce full spoken language sentences without additional training or fine-tuning.

To accommodate longer input sequences, we use a **sliding window** approach, where input pose sequences are divided into overlapping segments of fixed length. Each segment is processed independently, and outputs from multiple segments are combined to produce a final sentence-level prediction.

To do so, we use a pre-trained Transformer model to convert our **V** dimensional tokens of length *N* to a meaningful sentence.

Input data consists of preprocessed pose sequences paired with corresponding sentence-level annotations from a sign language corpus.

While the model is fixed, we explore the impact of several hyperparameters during inference:
- Window size (w)
- Window stride (s)
- Decoding strategy

## Pipeline Descriptions

We consider two main pipelines:

### 1. Baseline Pipeline (Greedy Concatenation)

- Long sentence level sign videos are segmented using a **sliding window** procedure.
- For each window, **SignCLIP** predicts the **most probable gloss/text**.
- The predicted words from each window are **concatenated** to form a sentence.

### 2. Transformer Pipeline (Top-k + Language Model)

- Each segment is again passed through **SignCLIP** to extract the **top-10 highest probable glosses**.
- A **Transformer-based language model** is used to generate coherent sentence-level outputs from these gloss sets using methods like **beam search**.

# Evaluation Protocol

We follow a consistent evaluation protocol across all experiments:

- Input-output pairs are drawn from a sentence-level annotated dataset, [How2Sign](https://how2sign.github.io/).
- Pose sequences are split into segments using fixed-length windows with overlap.
- The outputs from all segments are merged into a complete sentence.

We evaluate the generated sentences against ground-truth references using one of the following standard metrics for text generation:

- **BLEU**: Measures n-gram precision between predicted and reference sentences.
- **ROUGE-L**: Evaluates the longest common subsequence for recall-oriented matching.

