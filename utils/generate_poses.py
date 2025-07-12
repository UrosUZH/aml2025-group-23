import pandas as pd
import json
import subprocess
from pathlib import Path
import argparse

def convert_csv_to_json_with_ids(input_csv: Path, output_json: Path, id_start: int = 410) -> None:
    """
    Convert a How2Sign TSV CSV file to JSON format with incremental IDs.

    Args:
        input_csv (Path): Path to the input TSV CSV file.
        output_json (Path): Path to save the output JSON file.
        id_start (int): Starting ID value. Default is 410.

    Returns:
        None
    """
    columns = [
        "VIDEO_ID", "VIDEO_NAME", "SENTENCE_ID",
        "SENTENCE_NAME", "START", "END", "SENTENCE"
    ]

    df = pd.read_csv(
        input_csv,
        sep="\t",
        names=columns,
        quoting=3,
        encoding="utf-8",
        on_bad_lines="skip",
        header=0
    )
    print(df.head())

    # Drop rows with missing START or END
    df = df.dropna(subset=["START", "END"])
    df["START"] = df["START"].astype(float)
    df["END"] = df["END"].astype(float)
    df["SENTENCE"] = df["SENTENCE"].astype(str).str.strip()

    # Add incremental IDs
    df = df.reset_index(drop=True)
    df["id"] = df.index + id_start

    records = df[["id", "SENTENCE_NAME", "START", "END", "SENTENCE"]].rename(
        columns={
            "SENTENCE_NAME": "video_name",
            "START": "start",
            "END": "end",
            "SENTENCE": "sentence"
        }
    ).to_dict(orient="records")

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"✅ Successfully wrote {len(records)} entries to {output_json}")

def generate_pose_files(mock_dir: Path) -> None:
    """
    Generate pose files from MP4 videos in a given directory using video_to_pose.

    Args:
        mock_dir (Path): Directory containing MP4 files.

    Returns:
        None
    """
    mp4_files = list(mock_dir.glob("*.mp4"))
    if not mp4_files:
        print("⚠️ No MP4 files found.")
        return

    for mp4 in mp4_files:
        pose_file = mock_dir / f"{mp4.stem}.pose"
        if pose_file.exists():
            print(f"✅ Pose file already exists: {pose_file.name}")
            continue

        print(f"🔹 Creating pose for: {mp4.name}")
        cmd = [
            "video_to_pose",
            "-i", str(mp4),
            "--format", "mediapipe",
            "-o", str(pose_file)
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Created: {pose_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create pose for {mp4.name}: {e}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Process How2Sign CSV and generate pose files.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input TSV CSV file")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save output JSON file")
    parser.add_argument("--mock_dir", type=str, required=True, help="Directory containing MP4 videos")

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_json = Path(args.output_json)
    mock_dir = Path(args.mock_dir)

    convert_csv_to_json_with_ids(input_csv, output_json)
    generate_pose_files(mock_dir)

if __name__ == "__main__":
    main()
