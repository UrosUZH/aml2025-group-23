import argparse
import pandas as pd
from transformers import BertTokenizer
from pathlib import Path

def extract_vocab_from_csvs(
    csv_paths: list[Path],
    sentence_column: str = "SENTENCE",
    tokenizer_name: str = "bert-base-uncased",
) -> list[str]:
    """
    Extract unique tokens from a list of CSV files using a specified tokenizer.

    Args:
        csv_paths (list[Path]): List of CSV file paths.
        sentence_column (str): Column name containing sentences. Default is "SENTENCE".
        tokenizer_name (str): Name of Hugging Face tokenizer. Default is "bert-base-uncased".

    Returns:
        list[str]: Sorted list of unique tokens.
    """
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    vocab_set = set()

    for csv_file in csv_paths:
        print(f"Processing: {csv_file}")
        df = pd.read_csv(csv_file, sep="\t")

        if sentence_column in df.columns:
            sentences = df[sentence_column].tolist()
            for sentence in sentences:
                tokens = tokenizer.tokenize(sentence)
                vocab_set.update(tokens)
        else:
            print(f"⚠️ Warning: '{sentence_column}' column not found in {csv_file}")

    return sorted(vocab_set)

def save_vocab(vocab_list: list[str], output_path: Path) -> None:
    """
    Save a list of tokens to a text file.

    Args:
        vocab_list (list[str]): List of tokens.
        output_path (Path): Path to output vocab file.

    Returns:
        None
    """
    with output_path.open("w", encoding="utf-8") as f:
        for token in vocab_list:
            f.write(token + "\n")
    print(f"✅ Done! Vocab size: {len(vocab_list)}. Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract vocab from CSV sentence columns using a tokenizer.")
    parser.add_argument("--csvs", nargs="+", required=True, help="List of CSV file paths")
    parser.add_argument("--output", type=str, required=True, help="Path to output vocab text file")
    parser.add_argument("--sentence_column", type=str, default="SENTENCE", help="Sentence column name (default: SENTENCE)")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased", help="Tokenizer name (default: bert-base-uncased)")

    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csvs]
    vocab_list = extract_vocab_from_csvs(csv_paths, args.sentence_column, args.tokenizer)
    save_vocab(vocab_list, Path(args.output))

if __name__ == "__main__":
    main()
