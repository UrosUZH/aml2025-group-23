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

def run():
    pass


if __name__ == '__main__':
    run()