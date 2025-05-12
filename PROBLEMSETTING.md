# Problem Setting

We address the task of translating sign language pose sequences into natural language sentences. Formally, let  
**x = ⟨x₁, x₂, ..., x_T⟩**,  
where each **xᵢ ∈ ℝ^D** denotes a D-dimensional vector representing human pose features at time step *i*. Given such an input sequence, the goal is to generate a sequence  
**y = ⟨y₁, y₂, ..., y_N⟩**,  
where each **yⱼ** is a token from a fixed output vocabulary **V**, and *N* is the length of the generated sentence. This defines a sequence-to-sequence mapping problem from pose-based input to natural language output.

Let **f: ℝ^{T×D} → V^N** denote the model used for this mapping. We investigate whether a pretrained model **f**, initially trained to associate pose data with single-word glosses, can generalize to sentence-level generation in a **zero-shot** setting.

# Approach

We use a pretrained model based on the SignCLIP architecture. While SignCLIP was trained to align isolated signs with textual labels, we aim to evaluate its ability to produce full spoken language sentences without additional training or fine-tuning.

To accommodate longer input sequences, we use a **sliding window** approach, where input pose sequences are divided into overlapping segments of fixed length. Each segment is processed independently, and outputs from multiple segments are combined to produce a final sentence-level prediction.

Input data consists of preprocessed pose sequences paired with corresponding sentence-level annotations from a sign language corpus. The corpus includes signs from a natural language (e.g., ASL or another English-language sign dataset), but we do not constrain our study to any single language.

# Evaluation Protocol

We follow a consistent evaluation protocol across all experiments:

- Input-output pairs are drawn from a sentence-level annotated dataset.
- All evaluations are conducted in a **zero-shot** mode, using the pretrained model without any task-specific fine-tuning.
- Pose sequences are split into segments using fixed-length windows with overlap.
- Each segment is decoded into natural language using standard decoding methods (e.g., greedy or beam search).
- The outputs from all segments are merged into a complete sentence.

We evaluate the generated sentences against ground-truth references using standard metrics for text generation:

- **BLEU**: Measures n-gram precision between predicted and reference sentences.
- **ROUGE-L**: Evaluates the longest common subsequence for recall-oriented matching.
- **WER (Word Error Rate)**: Quantifies the minimum number of edits required to match the reference.

These metrics offer complementary insights into the accuracy and fluency of generated sentences.

# Hyperparameter Settings

While the model is fixed, we explore the impact of several hyperparameters during inference:

- **Window size (w)**: The length of each input segment.
- **Window stride (s)**: The amount of overlap between consecutive windows.
- **Decoding strategy**: Greedy decoding vs. beam search with beam width ∈ {3, 5}.

Hyperparameter values are selected based on performance on a held-out validation set.

# Goal

The primary goal is to explore whether pretrained sign-to-text models, specifically those trained at the gloss level, can generalize to sentence-level generation in a zero-shot setup. This serves as a step toward understanding the limits and capabilities of current pretrained models in bridging pose-based sign language data and spoken language output.
