import pandas as pd
import json
from pathlib import Path

def random_function():
    
    # Input and output paths
    input_csv = Path("mock_videos/how2sign_val.csv")
    output_json = Path("mock_videos/how2sign_val.json")
    mock_dir = Path("mock_videos")
    # Read the TSV file
    # Some rows have fewer than 7 columns, so we set names manually
    columns = [
        "VIDEO_ID",
        "VIDEO_NAME",
        "SENTENCE_ID",
        "SENTENCE_NAME",
        "START",
        "END",
        "SENTENCE"
    ]

    # Use error_bad_lines=False to skip problematic lines gracefully
    df = pd.read_csv(
        input_csv,
        sep="\t",
        names=columns,
        quoting=3,
        encoding="utf-8",
        on_bad_lines="skip", header=0
    )
    print(df.head())
    # Drop rows with missing START or END
    df = df.dropna(subset=["START", "END"])

    # Convert START and END to floats
    df["START"] = df["START"].astype(float)
    df["END"] = df["END"].astype(float)

    # Trim whitespace in SENTENCE
    df["SENTENCE"] = df["SENTENCE"].astype(str).str.strip()

    # Reset index to create clean IDs starting from 410
    df = df.reset_index(drop=True)
    df["id"] = df.index + 410

    # Build list of dictionaries
    records = df[["id", "SENTENCE_NAME", "START", "END", "SENTENCE"]].rename(
        columns={
            "SENTENCE_NAME": "video_name",
            "START": "start",
            "END": "end",
            "SENTENCE": "sentence"
        }
    ).to_dict(orient="records")

    # Write JSON
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"✅ Successfully wrote {len(records)} entries to {output_json}")

    mp4_files = list(mock_dir.glob("*.mp4"))
    print(mp4_files)
import sys
# sys.path.insert(1, '/home/signclip/fairseq/examples/MMPT')




import subprocess
from pathlib import Path

def generate_pose_files(mock_dir: Path):
    sys.path.insert(1, '/home/signclip/fairseq/examples/MMPT')
    mp4_files = list(mock_dir.glob("*.mp4"))
    if not mp4_files:
        print("No MP4 files found.")
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


if __name__ == "__main__":
    mock_dir = Path("mock_videos")
    random_function()
    generate_pose_files(mock_dir)
