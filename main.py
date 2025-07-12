from pathlib import Path
import sys
import os
import argparse

# Insert MMPT path
sys.path.insert(1, '/home/signclip/fairseq/examples/MMPT')
os.chdir('/home/signclip/fairseq/examples/MMPT')

from src.Transformer import TransformerDecoder
import src.Extraction as Extraction
import utils.utils as utils
from aml.src.sliding_window import embed_and_score_pose_segments_pipe
import aml.utils.generate_poses as generate_poses
from aml.utils.config_parser import load_config

def main():
    parser = argparse.ArgumentParser(description="Run sign language pipeline with optional mock mode and config.")
    parser.add_argument("--mock", dest="do_mock", action="store_true", help="Use mock sentences instead of actual decoding. (Sanity check mode)")
    parser.add_argument("--config", type=str, default="aml/config/default.yaml", help="Path to YAML config file.")
    args = parser.parse_args()

    # Load config
    config = load_config(Path(args.config))

    dataset_dir = Path(config["dataset_dir"])
    results_dir = Path(config["results_dir"])
    signclip_dir = results_dir / "signclip"
    eval_dir = results_dir / "evaluation"

    results_dir.mkdir(parents=True, exist_ok=True)
    signclip_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Derived CSV paths
    compare_csv_path = eval_dir / "sentences_compare.csv"
    augmented_csv_path = eval_dir / "sentences_compare_augmented.csv"
    evaluated_csv_path = eval_dir / "sentences_compare_evaluated.csv"

    vocab_embed_dir = Path(config["vocab_embed_dir"])
    vocab_text_path = Path(config["vocab_text_path"])
    alignment_csv_path = Path(config["alignment_csv_path"])

    beam_size = config["beam_size"]
    window_size = config["window_size"]
    stride = config["stride"]
    model_name = config["model_name"]
    top_k_similar = config["top_k_similar"]
    top_k_decode = config["top_k_decode"]

    # Load vocab embeddings
    text_embeddings, vocab_words = Extraction.get_vocab_embeddings(vocab_text_path, vocab_embed_dir)
    print(f"✅ Loaded {len(text_embeddings)} text embeddings and {len(vocab_words)} vocab words.")

    # Generate pose files if needed
    generate_poses.generate_pose_files(dataset_dir)
    pose_files = list(dataset_dir.glob("*.pose"))

    decoder = TransformerDecoder(
        lm_name="microsoft/phi-4-mini-instruct",
        beam_size=beam_size,
        alpha=0.95,
        beta=0.05,
        device="cuda",
        load_in_8bit=False,
        refine=False,
        debug=False,
    )

    for pose_path in pose_files:
        print(f"🔍 Processing: {pose_path.name}")

        csv_filename = f"{pose_path.stem}_W{window_size}_S{stride}.csv"
        output_csv_path = signclip_dir / csv_filename

        # Embed and score slices, save top_k_similar predictions
        embed_and_score_pose_segments_pipe(
            pose_path=pose_path,
            vocab_embed_path=vocab_embed_dir,
            vocab_text_path=vocab_text_path,
            output_csv_path=output_csv_path,
            window_size=window_size,
            stride=stride,
            model_name=model_name,
            k=top_k_similar
        )

        sentence_name = pose_path.stem

        # Load only top_k_decode candidates for decoding
        candidate_lists, _ = utils.load_top_candidates(output_csv_path, k=top_k_decode)

        if utils.should_skip_decoding(compare_csv_path, output_csv_path, sentence_name, beam_size, top_k=top_k_decode):
            print("⚠️ Skipping decoding — result already exists.")
            continue

        if args.do_mock:
            sentence = "this is a mock sentence"
            print("💡 Using mock sentence instead of real decoding.")
        else:
            decoder.scoring_mode = "default"
            sentence = decoder.decode(candidate_lists)

        utils.create_sentence_comparison_csv(
            output_csv_path,
            sentence_name,
            sentence,
            alignment_csv_path,
            compare_csv_path,
            beam_size=beam_size,
            k=top_k_decode
        )

    # Evaluation steps
    utils.augment_with_window_stride(
        input_csv_path=compare_csv_path,
        output_csv_path=augmented_csv_path
    )
    utils.evaluate_sentence_csv_rows(augmented_csv_path, evaluated_csv_path)
    utils.table_summary(evaluated_csv_path)

if __name__ == "__main__":
    main()
