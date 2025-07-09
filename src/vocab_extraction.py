import pandas as pd
from transformers import BertTokenizer

# Folder containing your CSV files
csv_folder = "mock_sentences/"

# Explicitly specify only the correct files
csv_files = [
    f"{csv_folder}/how2sign_train.csv",
    f"{csv_folder}/how2sign_val.csv",
    f"{csv_folder}/how2sign_test.csv",
]

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Set to store tokens
vocab_set = set()

# Process each CSV
for csv_file in csv_files:
    print(f"Processing: {csv_file}")
    df = pd.read_csv(csv_file, sep="\t")
    
    if 'SENTENCE' in df.columns:
        sentences = df['SENTENCE'].tolist()
        for sentence in sentences:
            tokens = tokenizer.tokenize(sentence)
            vocab_set.update(tokens)
    else:
        print(f"Warning: SENTENCE column not found in {csv_file}")

# Convert set to sorted list
vocab_list = sorted(vocab_set)

# Save vocab to file
with open("vocab_new.txt", "w") as f:
    for token in vocab_list:
        f.write(token + "\n")

print(f"✅ Done! Vocab size: {len(vocab_list)}. Saved to vocab_new.txt")
