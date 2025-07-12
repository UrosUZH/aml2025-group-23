import requests
from bs4 import BeautifulSoup
import time
import argparse
from pathlib import Path

BASE_URL = "https://www.spreadthesign.com/en.us/search/"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

def fetch_glosses_for_word(word: str, headers: dict, timeout: int = 10) -> list[str]:
    """
    Fetch glosses for a given word from SpreadTheSign in both sentence and word classes.

    Args:
        word (str): The word to search for.
        headers (dict): HTTP request headers.
        timeout (int): Request timeout in seconds.

    Returns:
        list[str]: List of cleaned gloss strings.
    """
    glosses = []

    for cls in [1, 2]:  # 1: Sentences, 2: Words
        params = {"q": word, "cls": cls}
        try:
            response = requests.get(BASE_URL, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"⚠️ Error fetching {word} with cls={cls}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        result_divs = soup.find_all("div", class_="search-result js-addclass js-addclass-search-results")

        for div in result_divs:
            title_div = div.find("div", class_="search-result-title")
            if title_div:
                a_tag = title_div.find("a")
                if a_tag and a_tag.text.strip():
                    gloss_text_full = a_tag.text.strip()
                    if cls == 2:
                        lines = [line.strip() for line in gloss_text_full.split("\n") if line.strip()]
                        cleaned_gloss = lines[0] if lines else gloss_text_full
                    else:
                        cleaned_gloss = gloss_text_full

                    glosses.append(cleaned_gloss)

    return glosses

def process_vocab(
    input_vocab_path: Path,
    output_gloss_path: Path,
    headers: dict = DEFAULT_HEADERS,
    request_timeout: int = 10,
    sleep_between_requests: float = 0.25
) -> None:
    """
    Process each word in a vocab file, query SpreadTheSign, and write glosses to an output file.

    Args:
        input_vocab_path (Path): Path to input vocab file.
        output_gloss_path (Path): Path to output gloss file.
        headers (dict): HTTP request headers.
        request_timeout (int): Request timeout in seconds.
        sleep_between_requests (float): Delay between queries in seconds.

    Returns:
        None
    """
    with input_vocab_path.open("r", encoding="utf-8") as f:
        vocab_words = [line.strip() for line in f if line.strip()]

    with output_gloss_path.open("w", encoding="utf-8") as f_out:
        for i, word in enumerate(vocab_words):
            print(f"[{i+1}/{len(vocab_words)}] Querying: {word}")

            glosses = fetch_glosses_for_word(word, headers, timeout=request_timeout)

            for gloss in glosses:
                print(f"  Found gloss: {gloss}")
                f_out.write(gloss + "\n")

            time.sleep(sleep_between_requests)

    print(f"✅ Finished! Saved results to: {output_gloss_path}")

def main():
    parser = argparse.ArgumentParser(description="Fetch glosses from SpreadTheSign for vocab words.")
    parser.add_argument("--input", type=str, required=True, help="Path to input vocab file (one word per line).")
    parser.add_argument("--output", type=str, required=True, help="Path to output gloss file.")
    parser.add_argument("--timeout", type=int, default=10, help="Request timeout in seconds (default: 10).")
    parser.add_argument("--sleep", type=float, default=0.25, help="Sleep time between requests in seconds (default: 0.25).")

    args = parser.parse_args()

    process_vocab(
        input_vocab_path=Path(args.input),
        output_gloss_path=Path(args.output),
        request_timeout=args.timeout,
        sleep_between_requests=args.sleep
    )

if __name__ == "__main__":
    main()
