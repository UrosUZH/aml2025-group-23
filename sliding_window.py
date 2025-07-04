import os
from pathlib import Path
import numpy as np
import torch
import numpy as np
import pandas as pd
from pose_format import Pose
import sys
sys.path.insert(1, '/home/signclip/fairseq/examples/MMPT')
os.chdir('/home/signclip/fairseq/examples/MMPT')
from mmpt.models import MMPTModel
import src.Extraction as extraction2
import demo_sign as extraction
# If your functions are in another file, adjust the import accordingly
# from your_module import embed_pose, preprocess_pose
import pickle
import av
def segment_video_sliding_windows(
    video_path: Path,
    output_dir: Path,
    window_size: int = 64,
    stride: int = 16
):
    """
    Load an MP4 file, create sliding window segments by frames, and save each segment as a new MP4 file.
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

def embed_pose_sliding_windows(
    pose_path: Path,
    output_dir: Path,
    window_size: int = 64,
    stride: int = 16,
    model_name: str = "default"
):
    """
    Load a pose file, create sliding window segments, embed each, and save embeddings.
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
    k: int = 5,
    output_csv_name: str = "topk_results.csv"
):
    """
    For all slice embeddings in a directory, compute similarity to vocab embeddings,
    retrieve top-k matches, and save results to CSV.
    """
    # Load vocab embeddings and labels
    # vocab_embeddings = extraction2.load_embeddings(vocab_embed_dir)
    
    with open(vocab_embed_dir, 'rb') as f:
        vocab_embeddings = pickle.load(f)
    with open(Path("aml/src") / "vocab.txt", "r") as f:
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

import numpy as np
import csv

import json
import pandas as pd

import json
import pandas as pd
import unicodedata

def clean_label(text):
    if not isinstance(text, str):
        return str(text)
    # Normalize to NFKD form and remove combining characters
    normalized = unicodedata.normalize("NFKD", text)
    cleaned = "".join(c for c in normalized if not unicodedata.combining(c))
    return cleaned


def print_top1_predictions_with_sentences(
    csv_path,
    json_path,
    vide_name: str = "video_name"  # This is not used in the function
):
    """
    For each slice, print video name, top-1 prediction, and reference sentence.
    """
    print(vide_name)
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
    sentence = video_to_sentence.get(vide_name, "[Sentence not found]")
    words = sentence.split()
    num_words = len(words)
    for i, row in df.iterrows():
        slice_name = row["slice"]

       
        video_prefix = slice_name.split("_W")[0] if "_W" in slice_name else slice_name.split("/")[0]

        sentence = video_to_sentence.get(vide_name, "[Sentence not found]")
        if i < num_words:
            word = words[i]
        else:
            word = "[no word]"
        label_1 = clean_label(row["label_1"])
        label_2 = clean_label(row["label_2"])
        label_3 = clean_label(row["label_3"])

        print(f"{video_prefix:30s} | {label_1:20s} | {label_2:20s} | {label_3:20s} | {word}")


if __name__ == "__main__":
    # 279MO2nwC_E_4-2-rgb_front
    #  C4lsh3mZ4jU_9_10-5-rgb_front
    #   3xni-I6N3EY_2-1-rgb_front
    
    vid_dir = Path("aml/mock_videos/3xni-I6N3EY_2-1-rgb_front")
    vid_dir = Path("aml/mock_videos/C4lsh3mZ4jU_9_10-5-rgb_front")
    pose_file = vid_dir.with_suffix(".pose")
    video_file = vid_dir.with_suffix(".mp4")
   
    window_size = 64
    stride = 64
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
    # Score and save results
    slice_dir = output_dir
    vocab_embed_dir = Path("aml/src/vocab_embed")
    if slice_dir.exists():
        print(f"⚠️ Slice directory {slice_dir} exist.")
    score_pose_slices(
        slice_dir=slice_dir,
        vocab_embed_dir=vocab_embed_dir,
        k=5
    )
    
    print_top1_predictions_with_sentences(
        csv_path=slice_dir / "topk_results.csv",
        json_path=Path("aml/mock_videos/how2sign_val.json"),
        vide_name = vid_dir.name
    )