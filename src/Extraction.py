from pose_format import Pose
import numpy as np
import torch
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
import warnings
import datetime
import pickle
import os
import sys
sys.path.insert(1, '/home/signclip/fairseq/examples/MMPT')
os.chdir('/home/signclip/fairseq/examples/MMPT')
from mmpt.models import MMPTModel

# Suppress TensorFlow C++ backend logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Suppress GLOG messages from XLA/CUDA
os.environ['GLOG_minloglevel'] = '3'

warnings.filterwarnings("ignore")           # Suppress Python warnings

logging.getLogger('tensorflow').setLevel(
    logging.ERROR)  # Suppress TensorFlow Python logs

# Optionally, if using Hugging Face Transformers, you can suppress its logs too:
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    pass


# Model configurations (keep these unchanged)
model_configs = [
    ("default", "signclip_v1_1/baseline_temporal"),  # multilingual pretrained
    # fine-tuned on ASL Citizen
    ("asl_citizen", "signclip_asl/asl_citizen_finetune"),
    # fine-tuned on three ASL datasets
    ("asl_finetune", "signclip_asl/asl_finetune"),
]

# Cache for models that have been lazily initialized.
models = {}
model_name = 'default'


def get_model(model_name):
    """
    Lazily load the requested model based on model_name.
    If the model is already loaded, return it.
    Otherwise, find its config, load it, and cache it.
    """
    if model_name in models:
        return models[model_name]

    # Look up the configuration for the given model_name.
    config_path = None
    for m_name, cfg in model_configs:
        if m_name == model_name:
            config_path = cfg
            break

    if config_path is None:
        raise ValueError(f"Unknown model name: {model_name}")

    # Load the model, tokenizer, and aligner.
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


def preprocess_text(text, model_name="default"):
    model_info = get_model(model_name)
    aligner = model_info["aligner"]
    tokenizer = model_info["tokenizer"]

    caps, cmasks = aligner._build_text_seq(
        tokenizer(text, add_special_tokens=False)["input_ids"],
    )
    caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

    return caps, cmasks


def embed_text(text, model_name='default'):
    model_info = get_model(model_name)
    model = model_info['model']

    # Determine the placeholder dimension based on the model_name.
    if model_name == 'lip':
        placeholder_dim = 1377
    elif model_name == 'lip_only':
        placeholder_dim = 768
    else:
        placeholder_dim = 609

    # Ensure texts is a list.
    texts = text if isinstance(text, list) else [text]
    batch_size = len(texts)

    # Preprocess each text individually and store the results.
    caps_list = []
    cmasks_list = []
    for t in texts:
        caps, cmasks = preprocess_text(t, model_name)
        caps_list.append(caps)   # Each should have shape (1, 128)
        cmasks_list.append(cmasks)

    # Concatenate the individual results along the batch dimension.
    caps_batch = torch.cat(caps_list, dim=0)
    cmasks_batch = torch.cat(cmasks_list, dim=0)

    # Create dummy pose_frames with shape (batch_size, 1, placeholder_dim).
    pose_frames = torch.randn(batch_size, 1, placeholder_dim)

    # Run the model forward pass only once with the full batch.
    with torch.no_grad():
        output = model(pose_frames, caps_batch,
                       cmasks_batch, return_score=False)

    # Extract the pooled text embeddings and return as a NumPy array.
    embeddings = output['pooled_text'].cpu().numpy()
    return embeddings


# ===================================== Text processing =====================================


def preprocess_text(text, model_name="default"):
    model_info = get_model(model_name)
    aligner = model_info["aligner"]
    tokenizer = model_info["tokenizer"]

    caps, cmasks = aligner._build_text_seq(
        tokenizer(text, add_special_tokens=False)["input_ids"],
    )
    caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

    return caps, cmasks


def embed_text(text, model_name='default'):
    model_info = get_model(model_name)
    model = model_info['model']

    # Determine the placeholder dimension based on the model_name.
    if model_name == 'lip':
        placeholder_dim = 1377
    elif model_name == 'lip_only':
        placeholder_dim = 768
    else:
        placeholder_dim = 609

    # Ensure texts is a list.
    # TODO: No more support for list maybe? Only accept str => Add typing to the definition
    texts = text if isinstance(text, list) else [text]
    batch_size = len(texts)

    # Preprocess each text individually and store the results.
    caps_list = []
    cmasks_list = []
    for t in texts:
        caps, cmasks = preprocess_text(t, model_name)
        caps_list.append(caps)   # Each should have shape (1, 128)
        cmasks_list.append(cmasks)

    # Concatenate the individual results along the batch dimension.
    caps_batch = torch.cat(caps_list, dim=0)
    cmasks_batch = torch.cat(cmasks_list, dim=0)

    # Create dummy pose_frames with shape (batch_size, 1, placeholder_dim).
    pose_frames = torch.randn(batch_size, 1, placeholder_dim)

    # Run the model forward pass only once with the full batch.
    with torch.no_grad():
        output = model(pose_frames, caps_batch,
                       cmasks_batch, return_score=False)

    # Extract the pooled text embeddings and return as a NumPy array.
    # TODO: Change cpu to cuda
    embeddings = output['pooled_text'].cpu().numpy()
    return embeddings


# ===================================== Pose processing =====================================

import mediapipe as mp
mp_holistic = mp.solutions.holistic
FACEMESH_CONTOURS_POINTS = [str(p) for p in sorted(
    set(p for p_tup in mp_holistic.FACEMESH_CONTOURS for p in p_tup))]

caps_pose_embedding, cmasks_pose_embedding = preprocess_text('', model_name)


def load_pose(pose_path):
    with open(pose_path, 'rb') as f:
        buffer = f.read()
        pose = Pose.read(buffer)
        return pose


def pose_normalization_info(pose_header):
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return pose_header.normalization_info(p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                                              p2=("POSE_LANDMARKS", "LEFT_SHOULDER"))

    if pose_header.components[0].name == "BODY_135":
        return pose_header.normalization_info(p1=("BODY_135", "RShoulder"),
                                              p2=("BODY_135", "LShoulder"))

    if pose_header.components[0].name == "pose_keypoints_2d":
        return pose_header.normalization_info(p1=("pose_keypoints_2d", "RShoulder"),
                                              p2=("pose_keypoints_2d", "LShoulder"))

    raise ValueError(
        f"Could not parse normalization info, pose_header.components[0].name is {pose_header.components[0].name}. Expected one of (POSE_LANDMARKS,BODY_135,pose_keypoints_2d)")


def pose_hide_legs(pose):
    if pose.header.components[0].name == "POSE_LANDMARKS":
        point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
        # pylint: disable=protected-access
        points = [
            pose.header._get_point_index("POSE_LANDMARKS", side + "_" + n)
            for n in point_names
            for side in ["LEFT", "RIGHT"]
        ]
        pose.body.confidence[:, :, points] = 0
        pose.body.data[:, :, points, :] = 0
        return pose
    raise ValueError("Unknown pose header schema for hiding legs")


def preprocess_pose(pose, max_frames=None):
    pose = pose.get_components(
        ["POSE_LANDMARKS", "FACE_LANDMARKS",
            "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
        {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS},
    )

    pose = pose.normalize(pose_normalization_info(pose.header))
    pose = pose_hide_legs(pose)

    feat = np.nan_to_num(pose.body.data)
    feat = feat.reshape(feat.shape[0], -1)

    # e.g., torch.Size([1, frame count, 609])
    pose_frames = torch.from_numpy(np.expand_dims(feat, axis=0)).float()
    if max_frames is not None and pose_frames.size(1) > max_frames:
        print(
            f"pose sequence length too long ({pose_frames.size(1)}) longer than {max_frames} frames. Truncating")
        pose_frames = pose_frames[:, :max_frames, :]

    return pose_frames


def embed_pose(pose, model_name='default'):
    model_info = get_model(model_name)
    model = model_info['model']

    poses = pose if type(pose) == list else [pose]
    embeddings = []

    pose_frames_l = []
    for p in poses:
        pose_frames = preprocess_pose(p)
        pose_frames_l.append(pose_frames)
    pose_frames_l = torch.cat(pose_frames_l)

    batch_size = len(poses)

    with torch.no_grad():
        output = model(pose_frames_l,
                       caps_pose_embedding.repeat(batch_size, 1),
                       cmasks_pose_embedding.repeat(batch_size, 1),
                       return_score=False)
        embeddings.append(output['pooled_video'].cpu().numpy())

    return np.concatenate(embeddings)


def get_pose_embeddings(pose_path, length_frames, padding_frames):
    pose_raw = load_pose(pose_path)
    print(f'pose_raw: {np.shape(pose_raw)}')

    # pose_processed = preprocess_pose(pose_raw)
    # print(f'pose_processed: {np.shape(pose_processed)}')
    
    pose_embeddings = embed_pose([pose_raw])
    print(f'pose_embeddings: {np.shape(pose_embeddings)}')

    # TODO: Split pose into chunks
#     pose_splitted = []
#     for start in range(0, len(pose_processed), length_frames-padding_frames):
#         end = min(start + length_frames, len(pose_processed))
# 
#         pose_splitted.append(pose_processed[start:end])
# 
#         if end == len(pose_processed):
#             break
# 
#     pose_embeddings = [embed_pose(pose, model_name) for pose in pose_splitted]
#     print(np.shape(pose_embeddings))
#     return pose_embeddings

#     model_info = get_model(model_name)
#     model = model_info["model"]
#
#     with torch.no_grad():
#         output = model(pose_frames, caps, cmasks, return_score=True)

    return pose_embeddings


# ===================================== Vocab =====================================


def create_vocab_embeddings(vocab_words, vocab_emb_path):
    print(f"Creating vocab embeddings - Start: {datetime.datetime.now()}")

    embeddings = [
        embed_text(word) for word in tqdm(vocab_words)
    ]

    embeddings = np.squeeze(embeddings, axis=1)

    with open(vocab_emb_path, 'wb') as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Creating vocab embeddings - Ended: {datetime.datetime.now()}")
    return embeddings


def load_vocab_embeddings(vocab_emb_path):
    print(f"Loading vocab embeddings - Start: {datetime.datetime.now()}")

    with open(vocab_emb_path, 'rb') as f:
        embeddings = pickle.load(f)

    print(f"Loading vocab embeddings - Ended: {datetime.datetime.now()}")
    return embeddings


def get_vocab_embeddings(
    # TODO
    vocab_words_path='/home/signclip/fairseq/examples/MMPT/aml/src/vocab.txt',
    vocab_emb_path='/home/signclip/fairseq/examples/MMPT/aml/src/vocab_embed',
    # vocab_words_path='/home/signclip/fairseq/examples/MMPT/aml/src/vocab_small.txt',
    # vocab_emb_path='/home/signclip/fairseq/examples/MMPT/aml/src/vocab_small_embed',
    tag_prompt="<en> <ase> "
):
    with open(vocab_words_path, 'r') as f:
        vocab_words = [f"{tag_prompt}{i.strip()}" for i in f.readlines()]

    if not os.path.exists(vocab_emb_path):
        embeddings = create_vocab_embeddings(vocab_words, vocab_emb_path)
    else:
        embeddings = load_vocab_embeddings(vocab_emb_path)

    return (embeddings, vocab_words)


# ===================================== Combined =====================================


def get_combined_scores(pose_path, model_name='default'):
    text_embeddings = get_vocab_embeddings()
    pose_embedding = get_pose_embeddings(pose_path, model_name)

    scores = np.matmul(pose_embedding, text_embeddings.T)
    return scores


# ===================================== Testing =====================================


def test():
    print(f"> start: {datetime.datetime.now()}")
    video_path = '/home/signclip/fairseq/examples/MMPT/zifan_A.mediapipe.mp4'
    pose_path = '/home/signclip/fairseq/examples/MMPT/aml/src/test.mediapipe.pose'

    print(f"> 1: {datetime.datetime.now()}")
    if not os.path.exists(pose_path):
        video_to_pose(video_path, pose_path)

    print(f"> 2: {datetime.datetime.now()}")
    pose_embeddings = get_pose_embeddings(pose_path, 10, 2)
    print(f'pose_embeddings: {np.shape(pose_embeddings)}')

    print(f"> 3: {datetime.datetime.now()}")
    text_embeddings, vocab_words = get_vocab_embeddings()
    print(f'text_embeddings: {np.shape(text_embeddings)}')

    print(f"> 4: {datetime.datetime.now()}")
    # with torch.no_grad():
        # output = model(pose_frames, caps, cmasks, return_score=True)

    scores = np.matmul(pose_embeddings, text_embeddings.T)
    print(f'scores: {np.shape(scores)}')
    vocab_scores = sorted([(scores[0][i], vocab_words[i]) for i in range(scores.shape[1])], reverse=True)
    print(vocab_scores[:10])

    print(f"> 5: {datetime.datetime.now()}")
    # os.remove(pose_path)

    print(f"> end: {datetime.datetime.now()}")


# ===================================== Main =====================================


if __name__ == '__main__':
    if sys.argv[1] == 'vocab':
        embeddings = get_vocab_embeddings()
        print(np.shape(embeddings[1]))
    elif sys.argv[1] == 'test':
        test()
