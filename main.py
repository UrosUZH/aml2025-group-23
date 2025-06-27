'''

Steps:
- Extract the pose from video
- Load the pose
- Create pose segments (with stride and windows size)
- Create embeddings for each pose
- Load the pre-computed embeddings for vocab words
- Calculate the similarity between the embeddings
- Sort and select top 10 predictions
- Apply soft max on the output
- Give it to the transformer model

<transformer => Roham>
input:
- an array of top 10 vocab predictions
output:
- an array of sentence pairs (truth, predicted)

<evaluation => Uros>
input:
- an array of sentence pairs (truth, predicted)
output:
- Plots

'''

# TODO: Step 1: Load video
# TODO: Step 2: Pass it to extraction to predictions
# TODO: Step 3: Pass it to Transformer to predict the sentence
# TODO: Step 4: Pass the sentences to Evaluation and get comparison result
# TODO: Step 5: Show plots, etc if needed

#! How to Import
# import sys
# sys.path.insert(1, '/home/signclip/fairseq/examples/MMPT')
# from mmpt.models import MMPTModel
# import os
# os.chdir('/home/signclip/fairseq/examples/MMPT')


# EVALUATION
import src.Evaluation as Evaluation
from src.Evaluation import evaluate, make_sentences
def run_mock_evaluation():
    make_sentences()
    evaluate(
        reference_path="mock_sentences/reference.txt",
        hypothesis_paths=[
            "mock_sentences/shuffled.txt",
            "mock_sentences/rand_03.txt",
            "mock_sentences/rand_05.txt",
            "mock_sentences/rand_09.txt"
        ],
        labels=["Reference", "Shuffled", "Noisy 30%", "Noisy 50%", "Noisy 90%"],
        use_bert=False,
        use_cosine=False
    )





def run():
    pass



if __name__ == '__main__':
    run_mock_evaluation()
    run()