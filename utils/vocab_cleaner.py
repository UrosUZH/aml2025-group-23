import argparse
import nltk
import geonamescache
import re
from nltk.corpus import words, names

# Download necessary NLTK data
nltk.download("words")
nltk.download("names")

# Prepare word sets
english_words = set(words.words())
name_words = set(names.words())

# Geo names cache for cities
gc = geonamescache.GeonamesCache()
city_names = set(city["name"].lower() for city in gc.get_cities().values())

def is_clean_token(token: str) -> bool:
    """
    Check if a token is a valid clean token based on English words, names, or city names.

    Args:
        token (str): Token to check.

    Returns:
        bool: True if clean, False otherwise.
    """
    # Ignore subwords (like ##ing)
    if token.startswith("##"):
        return False
    # Ignore purely non-alphabetic tokens
    if not re.fullmatch(r"[a-zA-Z]+", token):
        return False
    # Lowercase for consistency
    t = token.lower()
    return t in english_words or t in name_words or t in city_names

def filter_vocab(input_path: str, output_path: str) -> int:
    """
    Filter tokens in a vocab file and save cleaned tokens to a new file.

    Args:
        input_path (str): Path to input vocab file.
        output_path (str): Path to output cleaned vocab file.

    Returns:
        int: Number of clean tokens saved.
    """
    special_tokens = {"[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"}
    clean_vocab = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token in special_tokens:
                continue
            if is_clean_token(token):
                clean_vocab.append(token)

    with open(output_path, "w", encoding="utf-8") as f:
        for token in clean_vocab:
            f.write(token + "\n")

    return len(clean_vocab)

def main():
    parser = argparse.ArgumentParser(description="Filter a vocab file to only include clean English tokens and save to new file.")
    parser.add_argument("--input", type=str, required=True, help="Path to input vocab file")
    parser.add_argument("--output", type=str, required=True, help="Path to output cleaned vocab file")
    args = parser.parse_args()

    num_clean = filter_vocab(args.input, args.output)
    print(f"✅ Clean vocab saved to: {args.output}")
    print(f"✅ Clean vocab size: {num_clean}")

if __name__ == "__main__":
    main()
