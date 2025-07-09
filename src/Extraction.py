import sys
sys.path.insert(1, '/home/signclip/fairseq/examples/MMPT')
from mmpt.models import MMPTModel
import os
os.chdir('/home/signclip/fairseq/examples/MMPT')
import pickle
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress TensorFlow C++ backend logs
os.environ['GLOG_minloglevel'] = '3'          # Suppress GLOG messages from XLA/CUDA

import warnings
warnings.filterwarnings("ignore")           # Suppress Python warnings

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppress TensorFlow Python logs

# Optionally, if using Hugging Face Transformers, you can suppress its logs too:
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    pass

from tqdm import tqdm
import argparse
from pathlib import Path
import torch
import numpy as np
from pose_format import Pose

# Model configurations (keep these unchanged)
model_configs = [
    ("default", "signclip_v1_1/baseline_temporal"), # multilingual pretrained
    ("asl_citizen", "signclip_asl/asl_citizen_finetune"), # fine-tuned on ASL Citizen
    ("asl_finetune", "signclip_asl/asl_finetune"), # fine-tuned on three ASL datasets
]

# Cache for models that have been lazily initialized.
models = {}

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
        output = model(pose_frames, caps_batch, cmasks_batch, return_score=False)
    
    # Extract the pooled text embeddings and return as a NumPy array.
    embeddings = output['pooled_text'].cpu().numpy()
    return embeddings

'''
# import mediapipe as mp
# mp_holistic = mp.solutions.holistic
# FACEMESH_CONTOURS_POINTS = [str(p) for p in sorted(set(p for p_tup in mp_holistic.FACEMESH_CONTOURS for p in p_tup))]

# To avoid installing mediapipe, we just hardcode the face contours given the above code
# FACEMESH_CONTOURS_POINTS = [
#     '0', '7', '10', '13', '14', '17', '21', '33', '37', '39', '40', '46', '52', '53', '54', '55', '58', '61', '63',
#     '65', '66', '67', '70', '78', '80', '81', '82', '84', '87', '88', '91', '93', '95', '103', '105', '107', '109',
#     '127', '132', '133', '136', '144', '145', '146', '148', '149', '150', '152', '153', '154', '155', '157', '158',
#     '159', '160', '161', '162', '163', '172', '173', '176', '178', '181', '185', '191', '234', '246', '249', '251',
#     '263', '267', '269', '270', '276', '282', '283', '284', '285', '288', '291', '293', '295', '296', '297', '300',
#     '308', '310', '311', '312', '314', '317', '318', '321', '323', '324', '332', '334', '336', '338', '356', '361',
#     '362', '365', '373', '374', '375', '377', '378', '379', '380', '381', '382', '384', '385', '386', '387', '388',
#     '389', '390', '397', '398', '400', '402', '405', '409', '415', '454', '466'
# ]
# 
# MAX_FRAMES_DEFAULT = 256  # Default truncate length, can be overridden

# def pose_normalization_info(pose_header):
#     if pose_header.components[0].name == "POSE_LANDMARKS":
#         return pose_header.normalization_info(p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
#                                               p2=("POSE_LANDMARKS", "LEFT_SHOULDER"))
# 
#     if pose_header.components[0].name == "BODY_135":
#         return pose_header.normalization_info(p1=("BODY_135", "RShoulder"),
#                                               p2=("BODY_135", "LShoulder"))
# 
#     if pose_header.components[0].name == "pose_keypoints_2d":
#         return pose_header.normalization_info(p1=("pose_keypoints_2d", "RShoulder"),
#                                               p2=("pose_keypoints_2d", "LShoulder"))
#     
#     raise ValueError(f"Could not parse normalization info, pose_header.components[0].name is {pose_header.components[0].name}. Expected one of (POSE_LANDMARKS,BODY_135,pose_keypoints_2d)")


# def pose_hide_legs(pose):
#     if pose.header.components[0].name == "POSE_LANDMARKS":
#         point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
#         # pylint: disable=protected-access
#         points = [
#             pose.header._get_point_index("POSE_LANDMARKS", side + "_" + n)
#             for n in point_names
#             for side in ["LEFT", "RIGHT"]
#         ]
#         pose.body.confidence[:, :, points] = 0
#         pose.body.data[:, :, points, :] = 0
#         return pose
#     raise ValueError("Unknown pose header schema for hiding legs")


# def preprocess_pose(pose, max_frames=None):
#     pose = pose.get_components(
#         ["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
#         {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS},
#     )
# 
#     pose = pose.normalize(pose_normalization_info(pose.header))
#     pose = pose_hide_legs(pose)
# 
#     feat = np.nan_to_num(pose.body.data)
#     feat = feat.reshape(feat.shape[0], -1)
# 
#     pose_frames = torch.from_numpy(np.expand_dims(feat, axis=0)).float()  # e.g., torch.Size([1, frame count, 609])
#     if max_frames is not None and pose_frames.size(1) > max_frames:
#         print(f"pose sequence length too long ({pose_frames.size(1)}) longer than {max_frames} frames. Truncating")
#         pose_frames = pose_frames[:, :max_frames, :]
# 
#     return pose_frames


# def embed_pose(pose, model_name='default'):
#     model_info = get_model(model_name)
#     model = model_info['model']
# 
#     caps, cmasks = preprocess_text('', model_name)
#     poses = pose if type(pose) == list else [pose]
#     embeddings = []
# 
#     pose_frames_l = []
#     for p in poses:
#         pose_frames = preprocess_pose(p)
#         pose_frames_l.append(pose_frames)
#     pose_frames_l = torch.cat(pose_frames_l)
# 
#     batch_size = len(poses)
# 
#     with torch.no_grad():
#         output = model(pose_frames_l,
#                        caps.repeat(batch_size, 1),
#                        cmasks.repeat(batch_size, 1),
#                        return_score=False)
#         embeddings.append(output['pooled_video'].cpu().numpy())
# 
#     return np.concatenate(embeddings)




# def score_pose_and_text(pose, text, model_name="default", max_frames=None):
#     model_info = get_model(model_name)
#     model = model_info["model"]
# 
#     pose_frames = preprocess_pose(pose, max_frames)
#     caps, cmasks = preprocess_text(text, model_name)
# 
#     with torch.no_grad():
#         output = model(pose_frames, caps, cmasks, return_score=True)
# 
#     return text, float(output["score"])  # dot-product


# def score_pose_and_text_batch(pose, text, model_name='default'):
#     pose_embedding = embed_pose(pose, model_name)
#     text_embedding = embed_text(text, model_name)
# 
#     scores = np.matmul(pose_embedding, text_embedding.T)
#     return scores


# def main():
#     parser = argparse.ArgumentParser(description="Evaluate pose and text similarity using SignCLIP.")
#     parser.add_argument(
#         "--pose_path",
#         default="./house.pose",
#         type=Path,
#         help="Path to the .pose file.",
#     )
#     parser.add_argument(
#         "--max_frames",
#         nargs="?",
#         type=int,
#         const=MAX_FRAMES_DEFAULT,
#         default=None,
#         help=f"If provided, pose sequences longer than this will be truncated, otherwise they will not. If provided without a value, will use {MAX_FRAMES_DEFAULT}, as SignCLIP can currently only support this many. If provided with a value, will use that value",
#     )
# 
#     args = parser.parse_args()
# 
#     pose_path = args.pose_path
#     max_frames = args.max_frames
# 
#     if not pose_path.is_file():
#         print(f"Error: File {pose_path} does not exist.")
#         return
# 
#     with open(pose_path, "rb") as f:
#         buffer = f.read()
#         pose = Pose.read(buffer)
# 
#         print(score_pose_and_text(pose, "random text", max_frames=max_frames))
#         print(score_pose_and_text(pose, "house", max_frames=max_frames))
#         print(score_pose_and_text(pose, "<en> <ase> house", max_frames=max_frames))
#         print(score_pose_and_text(pose, "<en> <gsg> house", max_frames=max_frames))
#         print(score_pose_and_text(pose, "<en> <fsl> house", max_frames=max_frames))
#         print(score_pose_and_text(pose, "<en> <ase> sun", max_frames=max_frames))
#         print(score_pose_and_text(pose, "<en> <ase> police", max_frames=max_frames))
#         print(score_pose_and_text(pose, "<en> <ase> how are you?", max_frames=max_frames))
# 
#         text_l = ["<en> <ase> house", "<en> <ase> police"]
#         pose_l = [pose, pose]
#         print(score_pose_and_text_batch(pose_l, text_l))
#         
#         print(score_pose_and_text_batch(pose_l, text_l, model_name='asl_finetune'))
'''

def get_vocab(vocab_filename='/home/signclip/fairseq/examples/MMPT/aml/src/vocab_new.txt', tag_prompt="<en> <ase>"):
    print(f"Loading vocab - Start: {datetime.datetime.now()}")
    with open(vocab_filename, 'r') as f:
        raw_vocab = [i.strip() for i in f.readlines()]
        # Add tag prompt
        vocab = [f"{tag_prompt} {word}" for word in raw_vocab]
    print(f"Loading vocab - Ended: {datetime.datetime.now()}")
    return vocab

def store_embeddings(embeddings, filename):
    print(f"Storing vocab embeddings - Start: {datetime.datetime.now()}")
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Storing vocab embeddings - Ended: {datetime.datetime.now()}")

def load_embeddings(filename):
    print(f"Loading vocab embeddings - Start: {datetime.datetime.now()}")
    with open(filename, 'rb') as f:
        embeddings = pickle.load(f)
        print(f"Loading vocab embeddings - Ended: {datetime.datetime.now()}")
        return embeddings

def create_vocab_embeddings():
    vocab_words = get_vocab()

    embeddings = []
    for word in tqdm(vocab_words):
        embeddings.append(embed_text(word, model_name='asl_finetune'))

    vocab_save_path =  '/home/signclip/fairseq/examples/MMPT/aml/src/vocab_embed'
    store_embeddings(embeddings, vocab_save_path)

    print(f"Finished {datetime.datetime.now()}")

def load_vocab_embeddings():
    vocab_save_path =  '/home/signclip/fairseq/examples/MMPT/aml/src/vocab_embed'
    return load_embeddings(vocab_save_path)

if __name__ == '__main__':
    if sys.argv[1] == 'create':
        create_vocab_embeddings()
    elif sys.argv[1] == 'load':
        embeddings = load_vocab_embeddings()
        print(np.shape(embeddings))
