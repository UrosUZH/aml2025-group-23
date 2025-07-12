# Advanced Machine Learning Project - FS25

## Team members

- **Uros Dimitrijevic**: uros.dimitrijevic@uzh.ch | 15-936-834
- **Roham Zendehdel Nobari**: roham.zendehdelnobari@uzh.ch | 23-755-432
- **Mohammad Mahdi Hejazi**: mohammadmahdi.hejazi@uzh.ch | 24-748-998

## **Running the evaluation in a Dev Container (VSCode)**

The easiest way to get started is to use **VSCode** together with the **Dev Containers** extension. This automates all setup steps using the repository’s Dockerfile.

---

### **Steps**

-  **Open VSCode** and install the extension:

> **Extension**: *Dev Containers* (by Microsoft)

- Press `Alt + Shift + P` (or `Cmd + Shift + P` on Mac) to open the command palette.

- Type **"Attach to Container"** and select:

## How to run
Once you’re inside the container, do:
```Shell
cd ~/fairseq/examples/MMPT/
```
and then, simply run:
```Shell
bash aml/bash_scripts/run_eval.sh
```

**NOTE**: There might be some packages that slipped our minds when assembling this project. If you encounter errors, run this command:
```Shell
bash aml/bash_scripts/setup.sh
```
or you can do:
```Shell
python3 setup.py
```
## AML Project Configuration (`aml/config/default.yaml`)

This YAML file defines the default settings for running the full sign language evaluation pipeline.

Below is a breakdown of each parameter:

---

### Dataset and output directories

- `dataset_dir`:  
  Path to the folder containing your input pose files (or where pose files will be generated from videos).  
  Example: `aml/data/how2sign_subset`

- `results_dir`:  
  Base directory where all outputs will be saved. This includes embeddings, intermediate CSVs, and evaluation results.  
  Example: `aml/results`

---

### Vocabulary configuration

- `vocab_embed_dir`:  
  Directory where pre-computed vocabulary embeddings (from text glosses) are stored or will be saved.  
  Example: `aml/data/vocab/gloss_vocab_embed`

- `vocab_text_path`:  
  Path to the vocabulary text file (e.g., list of gloss words).  
  Example: `aml/data/vocab/gloss_vocab.txt`

---

### Alignment file

- `alignment_csv_path`:  
  Path to the CSV file containing ground-truth reference sentences, used for evaluating the final predicted sentences.  
  Example: `aml/data/mock_sentences/how2sign_realigned_val.csv`

---

### Model and decoding parameters

- `beam_size`:  
  Number of beams to use during Transformer decoding. Controls how many hypotheses are kept at each decoding



## Project scope

- This is an advanced machine learning project
- it is an extension of     
- Goal is to generate sentences from ASL videos


use Connectionist Temporal Classification (CTC) same as they do in audio modern speech recognition.
https://dl.acm.org/doi/pdf/10.1145/3131672.3131693


poses can be segmented


## Pipeline draft

- Prepare pretrained models from SIGNCLIP
    - certain tokens only
    - only one language
    - minimize
- Prepare video dataset
    - extract pose from video
    - perform various segmentations (size/number)
- Run it through the model
    - retrieve probablity scores
    - find the intervals with highest probablilty scores
- Run it through some state of the art transformer bart/t5
    - generate most probable sentence
- Evaluate generated sentence based on true video sentence (BLEU/ROGUE)

![pipeline-image](<AML Pipeline.png>)


    