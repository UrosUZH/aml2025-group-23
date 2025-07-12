from os import chdir
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from pose_format import Pose
import pickle
import av
import pympi
import json
import unicodedata
import argparse
import sys
# Set the path to the MMPT examples directory
sys.path.insert(1, '/home/signclip/fairseq/examples/MMPT')
chdir('/home/signclip/fairseq/examples/MMPT')
import demo_sign as extraction

def read_sign_segments_from_eaf(eaf_path: Path) -> list[dict]:
    """
    Parse an ELAN EAF file to extract sign segments as frame indices.

    Args:
        eaf_path (Path): Path to the EAF annotation file.

    Returns:
        list[dict]: List of segments, each with keys 'start_frame' and 'end_frame'.
    """
    eaf = pympi.Elan.Eaf(str(eaf_path))
    fps = 25  # You can adjust if known; some How2Sign videos are 25 FPS

    sign_segments = []
    for annotation in eaf.get_annotation_data_for_tier("SIGN"):
        start_time_ms = annotation[0]
        end_time_ms = annotation[1]

        start_frame = int(start_time_ms / 1000 * fps)
        end_frame = int(end_time_ms / 1000 * fps)
        sign_segments.append({"start_frame": start_frame, "end_frame": end_frame})

    return sign_segments

# Check https://github.com/sign-language-processing/segmentation/tree/main for how to segment videos by sign segments. 
# I created a seperate conda environment for it's installation "pip install git+https://github.com/sign-language-processing/segmentation"
# DON'T INSTALL IT IN THE signclip_inf conda env or it will break stuff!!!

def segment_video_by_sign_segments(
    video_path: Path,
    output_dir: Path,
    sign_segments: list
):
    """
    Segment a video into multiple clips based on provided sign segments (frame ranges).

    Args:
        video_path (Path): Path to the input video file.
        output_dir (Path): Directory where segmented video files will be saved.
        sign_segments (list): List of dicts with 'start_frame' and 'end_frame' for each segment.

    Returns:
        None
    """
    if output_dir.exists():
        print(f"⚠️ Output directory {output_dir} already exists. Skipping creation.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    container = av.open(str(video_path))
    video_stream = container.streams.video[0]

    fps = video_stream.average_rate
    frames = list(container.decode(video=0))
    num_frames = len(frames)

    slice_count = 0

    for idx, seg in enumerate(sign_segments):
        start_idx = seg["start_frame"]
        end_idx = seg["end_frame"]
        if end_idx >= num_frames:
            end_idx = num_frames - 1

        segment_frames = frames[start_idx:end_idx+1]

        segment_filename = f"sign_{idx:03d}_{start_idx}_{end_idx}.mp4"
        segment_path = output_dir / segment_filename

        output = av.open(str(segment_path), mode="w")
        stream = output.add_stream("libx264", rate=fps)
        stream.width = video_stream.width
        stream.height = video_stream.height
        stream.pix_fmt = "yuv420p"

        for f in segment_frames:
            packet = stream.encode(f)
            if packet:
                output.mux(packet)

        packet = stream.encode(None)
        if packet:
            output.mux(packet)
        output.close()

        print(f"✅ Saved sign segment: {segment_filename}")
        slice_count += 1

    print(f"✅ Finished: {slice_count} sign segments saved in {output_dir}")


def segment_pose_by_sign_segments(
    pose_path: Path,
    output_dir: Path,
    sign_segments: list
):
    """
    Segment a pose file into multiple files based on sign-level frame ranges.

    Args:
        pose_path (Path): Path to the input pose file.
        output_dir (Path): Directory to save segmented pose files.
        sign_segments (list): List of dicts with 'start_frame' and 'end_frame' for each segment.

    Returns:
        None
    """
    # Load original pose
    with pose_path.open("rb") as f:
        pose = Pose.read(f)

    pose_data = pose.body.data
    num_frames = pose_data.shape[0]

    if output_dir.exists():
        print(f"⚠️ Pose output directory {output_dir} already exists. Skipping creation.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, seg in enumerate(sign_segments):
        start_idx = seg["start_frame"]
        end_idx = seg["end_frame"]
        if end_idx >= num_frames:
            end_idx = num_frames - 1

        segment_pose = Pose(header=pose.header, body=pose.body[start_idx:end_idx+1])
        segment_filename = f"sign_{idx:03d}_{start_idx}_{end_idx}.pose"
        segment_path = output_dir / segment_filename

        with segment_path.open("wb") as f:
            segment_pose.write(f)

        print(f"✅ Saved sign pose segment: {segment_filename}")



def segment_video_sliding_windows(
    video_path: Path,
    output_dir: Path,
    window_size: int = 64,
    stride: int = 16
):
    """
    Segment a video into overlapping sliding windows by frame count.

    Args:
        video_path (Path): Path to the input video file.
        output_dir (Path): Directory to save video slices.
        window_size (int): Number of frames per window.
        stride (int): Step size between windows.

    Returns:
        Path: Path to the output directory.
    """
    # Prepare output dir
    if output_dir.exists():
        print(f"⚠️ Output directory {output_dir} already exists. Skipping creation.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open video container
    container = av.open(str(video_path))
    video_stream = container.streams.video[0]

    num_frames = video_stream.frames
    fps = video_stream.average_rate
    print(f"Video FPS: {fps}, total frames: {num_frames}")

    slice_count = 0
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame)

    for start_idx in range(0, num_frames - window_size + 1, stride):
        end_idx = start_idx + window_size
        segment_frames = frames[start_idx:end_idx]

        segment_filename = f"slice_{start_idx:03d}_{end_idx:03d}.mp4"
        segment_path = output_dir / segment_filename

        output = av.open(str(segment_path), mode="w")
        stream = output.add_stream("libx264", rate=fps)
        stream.width = video_stream.width
        stream.height = video_stream.height
        stream.pix_fmt = "yuv420p"

        for f in segment_frames:
            packet = stream.encode(f)
            if packet:
                output.mux(packet)

        # Flush encoder
        packet = stream.encode(None)
        if packet:
            output.mux(packet)
        output.close()

        print(f"✅ Saved video segment: {segment_filename}")
        slice_count += 1

    if slice_count == 0:
        print("⚠️ No slices generated (video too short?).")
    else:
        print(f"✅ Finished: {slice_count} video segments saved in {output_dir}")
    return output_dir

def embed_full_pose_segment(
    pose_path: Path,
    output_dir: Path,
    model_name: str = "default"
):
    """
    Compute an embedding for an entire pose sequence and save it as a .npy file.

    Args:
        pose_path (Path): Path to the pose file.
        output_dir (Path): Directory to save the embedding.
        model_name (str): Name of the model to use for embedding.

    Returns:
        None
    """
    with pose_path.open("rb") as f:
        pose = Pose.read(f)

    # Preprocess the entire sequence
    pose_frames = extraction.preprocess_pose(pose)
    pose_frames = pose_frames.squeeze(0)

    # Wrap as (1, num_frames, feature_dim)
    pose_tensor = pose_frames.unsqueeze(0)

    # Embed
    with torch.no_grad():
        model_info = extraction.get_model(model_name)
        model = model_info["model"]
        caps, cmasks = extraction.preprocess_text("", model_name)
        output = model(
            pose_tensor,
            caps,
            cmasks,
            return_score=False
        )
        embedding = output["pooled_video"].cpu().numpy()

    # Save embedding
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_filename = f"{pose_path.stem}.npy"
    np.save(output_dir / emb_filename, embedding)

    print(f"✅ Saved full sign embedding: {emb_filename}")


def embed_pose_sliding_windows(
    pose_path: Path,
    output_dir: Path,
    window_size: int = 64,
    stride: int = 16,
    model_name: str = "default"
):
    """
    Segment a pose file into sliding window chunks, compute embeddings for each, and save.

    Args:
        pose_path (Path): Path to the input pose file.
        output_dir (Path): Directory to save embeddings.
        window_size (int): Number of frames per window.
        stride (int): Step size between windows.
        model_name (str): Name of the model to use for embedding.

    Returns:
        Path: Path to the output directory.
    """
    # Load pose
    with pose_path.open("rb") as f:
        buffer = f.read()
    pose = Pose.read(buffer)

    # Preprocess the entire pose sequence into normalized frames
    pose_frames = extraction.preprocess_pose(pose)  # shape: (1, num_frames, feature_dim)
    pose_frames = pose_frames.squeeze(0) # shape: (num_frames, feature_dim)

    num_frames = pose_frames.shape[0]
    feature_dim = pose_frames.shape[1]
    output_dir = output_dir
    # Prepare output directory
    
    if output_dir.exists():
        print(f"⚠️ Output directory {output_dir} already exists. Skipping creation.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    
    slice_count = 0

    for start_idx in range(0, num_frames - window_size + 1, stride):
        end_idx = start_idx + window_size

        # Slice the frames
        slice_frames = pose_frames[start_idx:end_idx, :]

        # Re-wrap as (1, window_size, feature_dim)
        slice_tensor = slice_frames.unsqueeze(0)

        # Embed
        with torch.no_grad():
            model_info = extraction.get_model(model_name)
            model = model_info["model"]
            caps, cmasks = extraction.preprocess_text("", model_name)
            output = model(
                slice_tensor,
                caps,
                cmasks,
                return_score=False
            )
            embedding = output["pooled_video"].cpu().numpy()

        # Save embedding
        slice_filename = f"slice_{start_idx:03d}_{end_idx:03d}.npy"
        np.save(output_dir / slice_filename, embedding)

        print(f"✅ Saved embedding for frames {start_idx}-{end_idx}: {slice_filename}")
        slice_count += 1

    if slice_count == 0:
        print(f"⚠️ No slices generated (sequence too short?).")

    else:
        print(f"✅ Finished: {slice_count} slices saved in {output_dir}")
    return output_dir
        
        
def score_pose_slices(
    slice_dir: Path,
    vocab_embed_dir: Path,
    vocab_text_path: Path,
    k: int = 5,
    output_csv_name: str = "topk_results.csv"
):
    """
    Compute similarity scores between pose slice embeddings and vocab embeddings, 
    retrieve top-k matches, and save to CSV.

    Args:
        slice_dir (Path): Directory containing slice .npy embeddings.
        vocab_embed_dir (Path): Path to the vocab embeddings pickle file.
        k (int): Number of top predictions to save per slice.
        output_csv_name (str): Output CSV filename.

    Returns:
        None
    """
    # Load vocab embeddings and labels
    # vocab_embeddings = extraction2.load_embeddings(vocab_embed_dir)
    
    with open(vocab_embed_dir, 'rb') as f:
        vocab_embeddings = pickle.load(f)
    with open(vocab_text_path, "r") as f:
        vocab = [line.strip() for line in f]

    print(np.shape(vocab_embeddings))
    records = []
    if isinstance(vocab_embeddings, list):
        vocab_embeddings = np.array(vocab_embeddings)
        vocab_embeddings = vocab_embeddings.squeeze(1)
    print(np.shape(vocab_embeddings))
    for npy_file in sorted(slice_dir.glob("*.npy")):
        embedding = np.load(npy_file).squeeze(0)
   
        scores = np.matmul(embedding, vocab_embeddings.T)

        # Top-k
        topk_idx = np.argsort(-scores)[:k]
        topk_labels = [vocab[i] for i in topk_idx]
        topk_scores = [scores[i] for i in topk_idx]

        record = {"slice": npy_file.name}
        for rank, (label, score) in enumerate(zip(topk_labels, topk_scores), start=1):
            record[f"label_{rank}"] = label
            record[f"score_{rank}"] = score
        records.append(record)

    # Make DataFrame
    df = pd.DataFrame(records)

    # Save CSV
    output_csv = slice_dir / output_csv_name
    df.to_csv(output_csv, index=False)

    print(f"✅ Top-{k} results saved to {output_csv}")

def clean_label(text):
    """
    Normalize text by removing accents and diacritics.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return str(text)
    # Normalize to NFKD form and remove combining characters
    normalized = unicodedata.normalize("NFKD", text)
    cleaned = "".join(c for c in normalized if not unicodedata.combining(c))
    return cleaned


def print_top1_predictions_with_sentences(
    csv_path,
    json_path,
    video_name: str = "video_name"
):
    """
    Print top-1 predicted labels alongside the ground-truth sentence words.

    Args:
        csv_path (Path): Path to CSV file containing slice predictions.
        json_path (Path): Path to JSON file containing reference sentences.
        video_name (str): Name of the video to match against reference sentences.

    Returns:
        None
    """
    print(video_name)
    # Load predictions CSV
    df = pd.read_csv(csv_path)
    print(df)
    # Load JSON annotations
    with open(json_path, "r") as f:
        annotations = json.load(f)

    # Build lookup: video name -> sentence
    video_to_sentence = {entry["video_name"]: entry["sentence"] for entry in annotations}

    # Print header
    print(f"{'Video Name':30s} | {'Top1 Label':20s} | {'Top1 Label':20s} | {'Top1 Label':20s} | Sentence")
    print("-" * 120)
    sentence = video_to_sentence.get(video_name, "[Sentence not found]")
    words = sentence.split()
    num_words = len(words)
    for i, row in df.iterrows():
        slice_name = row["slice"]

       
        video_prefix = slice_name.split("_W")[0] if "_W" in slice_name else slice_name.split("/")[0]

        sentence = video_to_sentence.get(video_name, "[Sentence not found]")
        if i < num_words:
            word = words[i]
        else:
            word = "[no word]"
        label_1 = clean_label(row["label_1"])
        label_2 = clean_label(row["label_2"])
        label_3 = clean_label(row["label_3"])

        print(f"{video_prefix:30s} | {label_1:20s} | {label_2:20s} | {label_3:20s} | {word}")


def embed_and_score_pose_segments_pipe(
    pose_path: Path,
    vocab_embed_path: Path,
    vocab_text_path: Path,
    output_csv_path: Path,
    window_size: int = 64,
    stride: int = 16,
    model_name: str = "default",
    k: int = 5,
):
    """
    End-to-end pipeline to segment a pose file into windows, embed each window,
    score against vocab embeddings, and save top-k results to CSV.

    Args:
        pose_path (Path): Path to the input pose file.
        vocab_embed_path (Path): Path to vocab embeddings pickle file.
        vocab_text_path (Path): Path to vocab text file.
        output_csv_path (Path): Path to output CSV file.
        window_size (int): Number of frames per window.
        stride (int): Step size between windows.
        model_name (str): Name of the model to use.
        k (int): Number of top predictions to save per slice.

    Returns:
        Path: Path to the output CSV file (if created).
    """
    if output_csv_path.exists():
        print(f"⚠️ Output CSV {output_csv_path} already exists. Skipping creation.")
        return output_csv_path
    # Load pose file
    with pose_path.open("rb") as f:
        buffer = f.read()
    print(f"🔍 Loading pose file: {pose_path}")
    pose = Pose.read(buffer)

    # Preprocess pose frames
    pose_frames = extraction.preprocess_pose(pose).squeeze(0)  # (num_frames, feature_dim)
    num_frames, feature_dim = pose_frames.shape

    # Load vocab embeddings + vocab labels
    with open(vocab_embed_path, 'rb') as f:
        vocab_embeddings = pickle.load(f)
    if isinstance(vocab_embeddings, list):
        vocab_embeddings = np.array(vocab_embeddings).squeeze(1)
    with open(vocab_text_path, 'r') as f:
        vocab = [line.strip() for line in f]

    print(f"✅ Vocab loaded: {len(vocab)} entries | Embeddings shape: {vocab_embeddings.shape}")

    # Load model once
    model_info = extraction.get_model(model_name)
    model = model_info["model"]
    caps, cmasks = extraction.preprocess_text("", model_name)

    records = []
    slice_count = 0

    for start_idx in range(0, max(num_frames - window_size, 0)+1, stride):
        
        end_idx = start_idx + window_size
        slice_frames = pose_frames[start_idx:end_idx, :]
        # print(f"🔍 Processing slice: {start_idx} to {end_idx} (shape: {slice_frames.shape})")
        # print(f"🔍 Processing slice: {start_idx} to {end_idx} (shape: {max(num_frames - window_size, 0)+1})")
        # if slice_frames.shape[0] < window_size:
        #     continue  # Skip short segments

        slice_tensor = slice_frames.unsqueeze(0)  # (1, window_size, feature_dim)

        # Embed
        with torch.no_grad():
            output = model(slice_tensor, caps, cmasks, return_score=False)
            embedding = output["pooled_video"].cpu().numpy().squeeze(0)  # (embed_dim,)

        # Score against vocab
        # scores = np.matmul(embedding, vocab_embeddings.T)  # (vocab_size,)
        # topk_idx = np.argsort(-scores)[:k]
        # topk_labels = [vocab[i] for i in topk_idx]
        # topk_scores = [scores[i] for i in topk_idx]
        
        
        scores = np.matmul(embedding, vocab_embeddings.T)  # (vocab_size,)
        sorted_idx = np.argsort(-scores)  # indices sorted by descending score

        topk_labels = []
        topk_scores = []

        for idx in sorted_idx:
            label = vocab[idx]
            score = scores[idx]
            if '##' in label:
                continue 
            if label not in topk_labels:
                topk_labels.append(label)
                topk_scores.append(score)
            if len(topk_labels) == k:
                break

        # Store result
        record = {"slice": f"slice_{start_idx:03d}_{end_idx:03d}.npy"}
        for rank, (label, score) in enumerate(zip(topk_labels, topk_scores), start=1):
            record[f"label_{rank}"] = label
            record[f"score_{rank}"] = score
        records.append(record)
        slice_count += 1

    # Final output
    if slice_count == 0:
        print("⚠️ No valid slices processed.")
    else:
        df = pd.DataFrame(records)
        df.to_csv(output_csv_path, index=False)
        print(f"✅ Scored {slice_count} slices → Saved to {output_csv_path}")
        return output_csv_path


def run_sign_segment_pipeline(
    vid_dir: Path,
    pose_file: Path,
    video_file: Path,
    eaf_file: Path,
    vocab_embed_dir: Path,
    vocab_text_path: Path,
    json_path: Path,
):
    """
    Run the full pipeline using sign segments from EAF.

    Args:
        vid_dir (Path): Base video directory.
        pose_file (Path): Pose file path.
        video_file (Path): Video file path.
        eaf_file (Path): EAF file path.
        vocab_embed_dir (Path): Path to vocab embeddings.
        json_path (Path): Path to JSON annotations.

    Returns:
        None
    """
    print("✅ Using sign segments from EAF")
    sign_segments = read_sign_segments_from_eaf(eaf_file)

    output_dir = Path(f"{vid_dir}_sign_segments")
    video_seg_dir = output_dir / "video_segments"
    pose_seg_dir = output_dir / "pose_segments"
    embed_dir = output_dir / "embeddings"

    # Segment video
    segment_video_by_sign_segments(
        video_path=video_file,
        output_dir=video_seg_dir,
        sign_segments=sign_segments
    )

    # Segment pose
    segment_pose_by_sign_segments(
        pose_path=pose_file,
        output_dir=pose_seg_dir,
        sign_segments=sign_segments
    )

    # Embed each segmented pose file
    if embed_dir.exists():
        print(f"⚠️ Embedding output directory {embed_dir} already exists. Skipping creation.")
    else:
        embed_dir.mkdir(parents=True, exist_ok=True)

    for pose_seg_file in sorted(pose_seg_dir.glob("*.pose")):
        print(f"🔹 Embedding full sign: {pose_seg_file.name}")
        embed_full_pose_segment(
            pose_path=pose_seg_file,
            output_dir=embed_dir,
            model_name="asl_finetune"
        )

    # Score each embedding
    score_pose_slices(
        slice_dir=embed_dir,
        vocab_embed_dir=vocab_embed_dir,
        vocab_text_path=vocab_text_path,
        k=10
    )

    # Print top-1 predictions
    print_top1_predictions_with_sentences(
        csv_path=embed_dir / "topk_results.csv",
        json_path=json_path,
        video_name=vid_dir.name
    )


def run_sliding_window_pipeline(
    vid_dir: Path,
    pose_file: Path,
    video_file: Path,
    vocab_embed_dir: Path,
    vocab_text_path: Path,
    json_path: Path,
    window_size: int,
    stride: int,
):
    """
    Run the full pipeline using sliding window segmentation.

    Args:
        vid_dir (Path): Base video directory.
        pose_file (Path): Pose file path.
        video_file (Path): Video file path.
        vocab_embed_dir (Path): Path to vocab embeddings.
        json_path (Path): Path to JSON annotations.
        window_size (int): Window size for sliding.
        stride (int): Stride size for sliding.

    Returns:
        None
    """
    print("✅ Using fixed sliding window approach")
    output_dir = Path(f"{vid_dir}_W{window_size}_S{stride}")

    embed_pose_sliding_windows(
        pose_path=pose_file,
        output_dir=output_dir,
        window_size=window_size,
        stride=stride,
        model_name="asl_finetune"
    )

    if do_vid_segment := True:
        segment_video_sliding_windows(
            video_path=video_file,
            output_dir=output_dir / "video_segments",
            window_size=window_size,
            stride=stride
        )

    slice_dir = output_dir
    score_pose_slices(
        slice_dir=slice_dir,
        vocab_embed_dir=vocab_embed_dir,
        vocab_text_path=vocab_text_path,
        k=10
    )

    print_top1_predictions_with_sentences(
        csv_path=slice_dir / "topk_results.csv",
        json_path=json_path,
        video_name=vid_dir.name
    )


## Main function to run the script for testing purposes

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process video and pose data with optional sign segments.")
    parser.add_argument("--vid_dir", type=str, required=True, help="Base video directory path (without extension).")
    parser.add_argument("--use_sign_segments", action="store_true", help="Use sign segments from EAF instead of sliding window.")
    parser.add_argument("--window_size", type=int, default=32, help="Window size for sliding window approach.")
    parser.add_argument("--stride", type=int, default=16, help="Stride for sliding window approach.")
    parser.add_argument("--eaf_file", type=str, help="Path to EAF file (required if using sign segments).")
    parser.add_argument("--vocab_embed_dir", type=str, default="aml/data/vocab/vocab_bert_embed", help="Path to vocab embedding directory.")
    parser.add_argument("--vocab_text_path", type=str, default="aml/data/vocab/vocab_bert.txt", help="Path to vocab text file (vocab.txt).")
    parser.add_argument("--json_path", type=str, default="aml/data/mock_videos/how2sign_val.json", help="Path to JSON annotations file.")

    args = parser.parse_args()

    vid_dir = Path(args.vid_dir)
    pose_file = vid_dir.with_suffix(".pose")
    video_file = vid_dir.with_suffix(".mp4")
    vocab_embed_dir = Path(args.vocab_embed_dir)
    vocab_text_path = Path(args.vocab_text_path)
    json_path = Path(args.json_path)

    if args.use_sign_segments:
        if not args.eaf_file:
            raise ValueError("EAF file path (--eaf_file) must be provided when using sign segments.")

        eaf_file = Path(args.eaf_file)
        run_sign_segment_pipeline(
            vid_dir=vid_dir,
            pose_file=pose_file,
            video_file=video_file,
            eaf_file=eaf_file,
            vocab_embed_dir=vocab_embed_dir,
            vocab_text_path=vocab_text_path,
            json_path=json_path,
        )
    else:
        run_sliding_window_pipeline(
            vid_dir=vid_dir,
            pose_file=pose_file,
            video_file=video_file,
            vocab_embed_dir=vocab_embed_dir,
            vocab_text_path=vocab_text_path,
            json_path=json_path,
            window_size=args.window_size,
            stride=args.stride,
        )
