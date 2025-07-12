import os
import sys
import argparse
import datetime
import pickle
import logging
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from pose_format import Pose

# Set MMPT paths
sys.path.insert(1, '/home/signclip/fairseq/examples/MMPT')
os.chdir('/home/signclip/fairseq/examples/MMPT')
from mmpt.models import MMPTModel

# ======================= Suppress logs =======================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.ERROR)
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    pass

# ======================= Model configurations =======================
model_configs = [
    ("default", "signclip_v1_1/baseline_temporal"),
    ("asl_citizen", "signclip_asl/asl_citizen_finetune"),
    ("asl_finetune", "signclip_asl/asl_finetune"),
]
models = {}

# Commented out as for now it's not used. Might use later. It also gives import errors!
# FACEMESH_CONTOURS_POINTS = [str(p) for p in sorted(set(p for p_tup in
#                           getattr(__import__('mediapipe.solutions.holistic', fromlist=['FACEMESH_CONTOURS']), 'FACEMESH_CONTOURS')
#                           for p in p_tup))]


# ======================= Load models =======================
def get_model(model_name: str = "default") -> dict:
    """
    Lazily load and return a SignCLIP model by name.

    Args:
        model_name (str): Name of the model configuration ('default', 'asl_citizen', or 'asl_finetune').

    Returns:
        dict: Dictionary containing the model, tokenizer, and aligner.
    """
    if model_name in models:
        return models[model_name]

    config_path = next((cfg for m_name, cfg in model_configs if m_name == model_name), None)
    if not config_path:
        raise ValueError(f"Unknown model name: {model_name}")

    model, tokenizer, aligner = MMPTModel.from_pretrained(
        f"projects/retri/{config_path}.yaml",
        video_encoder=None,
    )
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    models[model_name] = {
        "model": model,
        "tokenizer": tokenizer,
        "aligner": aligner,
    }
    return models[model_name]


# ======================= Text processing =======================
def preprocess_text(text: str, model_name: str = "default") -> tuple:
    """
    Tokenize and prepare text for input to the SignCLIP model.

    Args:
        text (str): Input text string.
        model_name (str): Model name for which to preprocess the text.

    Returns:
        tuple: Tokenized caps and cmasks tensors.
    """
    model_info = get_model(model_name)
    aligner = model_info["aligner"]
    tokenizer = model_info["tokenizer"]

    caps, cmasks = aligner._build_text_seq(
        tokenizer(text, add_special_tokens=False)["input_ids"]
    )
    caps, cmasks = caps[None, :], cmasks[None, :]
    return caps, cmasks


def embed_text(text: str, model_name: str = "default") -> np.ndarray:
    """
    Compute a text embedding using SignCLIP for a single string.

    Args:
        text (str): Input text string to embed.
        model_name (str): Model name to use for embedding.

    Returns:
        np.ndarray: Text embedding vector.
    """
    model_info = get_model(model_name)
    model = model_info['model']

    caps, cmasks = preprocess_text(text, model_name)
    pose_frames = torch.randn(1, 1, 609)

    with torch.no_grad():
        output = model(pose_frames, caps, cmasks, return_score=False)
    return output['pooled_text'].cpu().numpy()


# ======================= Pose processing =======================
def load_pose(pose_path: Path) -> Pose:
    """
    Load a pose file from a given path.

    Args:
        pose_path (Path): Path to the .pose file.

    Returns:
        Pose: Loaded Pose object.
    """
    with pose_path.open("rb") as f:
        return Pose.read(f.read())


def pose_normalization_info(pose_header):
    """
    Get normalization info (shoulder points) for pose based on header schema.

    Args:
        pose_header: Pose header object.

    Returns:
        tuple: Normalization info points for alignment.

    Raises:
        ValueError: If header schema is not recognized.
    """
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return pose_header.normalization_info(p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                                              p2=("POSE_LANDMARKS", "LEFT_SHOULDER"))
    if pose_header.components[0].name == "BODY_135":
        return pose_header.normalization_info(p1=("BODY_135", "RShoulder"),
                                              p2=("BODY_135", "LShoulder"))
    if pose_header.components[0].name == "pose_keypoints_2d":
        return pose_header.normalization_info(p1=("pose_keypoints_2d", "RShoulder"),
                                              p2=("pose_keypoints_2d", "LShoulder"))
    raise ValueError(f"Unknown pose header schema: {pose_header.components[0].name}")


def pose_hide_legs(pose: Pose) -> Pose:
    """
    Zero out data and confidence of leg points to focus on upper body and hands.

    Args:
        pose (Pose): Input Pose object.

    Returns:
        Pose: Modified Pose object with legs hidden.
    """
    if pose.header.components[0].name == "POSE_LANDMARKS":
        point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
        points = [
            pose.header._get_point_index("POSE_LANDMARKS", f"{side}_{name}")
            for name in point_names
            for side in ["LEFT", "RIGHT"]
        ]
        pose.body.confidence[:, :, points] = 0
        pose.body.data[:, :, points, :] = 0
        return pose
    raise ValueError("Unknown pose header schema for hiding legs")


def preprocess_pose(pose: Pose) -> torch.Tensor:
    """
    Preprocess pose: normalize, hide legs, flatten, and convert to tensor.

    Args:
        pose (Pose): Input Pose object.

    Returns:
        torch.Tensor: Preprocessed pose tensor of shape (1, num_frames, feature_dim).
    """
    pose = pose.get_components(
        ["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
        {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS},
    )
    pose = pose.normalize(pose_normalization_info(pose.header))
    pose = pose_hide_legs(pose)

    feat = np.nan_to_num(pose.body.data).reshape(pose.body.data.shape[0], -1)
    return torch.from_numpy(np.expand_dims(feat, axis=0)).float()


def embed_pose(pose: Pose, model_name: str = "default") -> np.ndarray:
    """
    Compute an embedding for a full pose sequence using SignCLIP.

    Args:
        pose (Pose): Input Pose object.
        model_name (str): Model name to use for embedding.

    Returns:
        np.ndarray: Pose embedding vector.
    """
    model_info = get_model(model_name)
    model = model_info['model']

    pose_frames = preprocess_pose(pose)

    caps, cmasks = preprocess_text('', model_name)
    batch_size = pose_frames.shape[0]

    with torch.no_grad():
        output = model(pose_frames, caps.repeat(batch_size, 1), cmasks.repeat(batch_size, 1), return_score=False)
    return output['pooled_video'].cpu().numpy()


# ======================= Vocab =======================
def create_vocab_embeddings(vocab_words: list[str], vocab_emb_path: Path) -> np.ndarray:
    """
    Create and save embeddings for a list of vocabulary words.

    Args:
        vocab_words (list of str): List of vocabulary words.
        vocab_emb_path (Path): Path to save the embeddings.

    Returns:
        np.ndarray: Array of vocabulary embeddings.
    """
    print(f"Creating vocab embeddings - Start: {datetime.datetime.now()}")
    embeddings = [embed_text(word) for word in tqdm(vocab_words)]
    embeddings = np.squeeze(embeddings, axis=1)

    with vocab_emb_path.open('wb') as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Creating vocab embeddings - End: {datetime.datetime.now()}")
    return embeddings


def load_vocab_embeddings(vocab_emb_path: Path) -> np.ndarray:
    """
    Load vocabulary embeddings from disk.

    Args:
        vocab_emb_path (Path): Path to embeddings file.

    Returns:
        np.ndarray: Loaded embeddings array.
    """
    print(f"Loading vocab embeddings - Start: {datetime.datetime.now()}")
    with vocab_emb_path.open('rb') as f:
        embeddings = pickle.load(f)
    print(f"Loading vocab embeddings - End: {datetime.datetime.now()}")
    return embeddings


def get_vocab_embeddings(vocab_words_path: Path, vocab_emb_path: Path, tag_prompt: str = "<en> <ase> ") -> tuple[np.ndarray, list[str]]:
    """
    Load or create vocabulary embeddings and associated words list.

    Args:
        vocab_words_path (Path): Path to vocabulary text file.
        vocab_emb_path (Path): Path to embeddings file to load/save.
        tag_prompt (str): Tag prefix to prepend to each vocab word.

    Returns:
        tuple: (embeddings array, list of vocab words with tags).
    """
    with vocab_words_path.open('r') as f:
        vocab_words = [f"{tag_prompt}{line.strip()}" for line in f]

    if not vocab_emb_path.exists():
        embeddings = create_vocab_embeddings(vocab_words, vocab_emb_path)
    else:
        embeddings = load_vocab_embeddings(vocab_emb_path)

    return embeddings, vocab_words


# ======================= CLI Main =======================
def main():
    parser = argparse.ArgumentParser(description="SignCLIP embedding and vocab pipeline")
    parser.add_argument("mode", choices=["vocab", "test"], help="Choose to generate vocab embeddings or run test.")
    parser.add_argument("--vocab_path", type=Path, default=Path("aml/data/vocab/vocab.txt"), help="Path to vocab text file.")
    parser.add_argument("--vocab_emb_path", type=Path, default=Path("aml/data/vocab/vocab_embed"), help="Path to save/load vocab embeddings.")
    args = parser.parse_args()

    if args.mode == "vocab":
        embeddings, vocab_words = get_vocab_embeddings(args.vocab_path, args.vocab_emb_path)
        print(f"✅ Created/loaded embeddings for {len(vocab_words)} words.")
    elif args.mode == "test":
        pose_path = Path("aml/src/test.mediapipe.pose")
        if not pose_path.exists():
            print("❌ Pose file not found. Please create it first.")
            return

        pose = load_pose(pose_path)
        pose_embedding = embed_pose(pose, model_name="default")

        embeddings, vocab_words = get_vocab_embeddings(args.vocab_path, args.vocab_emb_path)
        scores = np.matmul(pose_embedding, embeddings.T)

        vocab_scores = sorted([(score, vocab_words[i]) for i, score in enumerate(scores[0])], reverse=True)

        print("Top 10 matched vocabulary entries:")
        for score, word in vocab_scores[:10]:
            print(f"  {score:.4f}  -  {word}")

if __name__ == "__main__":
    main()
