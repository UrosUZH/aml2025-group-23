# Advanced Machine Learning Project - FS25

## Team members

- **Uros Dimitrijevic**: uros.dimitrijevic@uzh.ch | 15-936-834
- **Roham Zendehdel Nobari**: roham.zendehdelnobari@uzh.ch | 23-755-432
- **Mohammad Mahdi Hejazi**: mohammadmahdi.hejazi@uzh.ch | 24-748-998

## Project scope

- This is an advanced machine learning project
- it is an extension of     
- Goal is to generate sentences from ASL videos


use Connectionist Temporal Classification (CTC) same as they do in audio modern speech recognition.
https://dl.acm.org/doi/pdf/10.1145/3131672.3131693


poses can be segmented


## Pipeline draft

- Prepare pretrained models from SIGNCLIP
    - certain tokens only
    - only one language
    - minimize
- Prepare video dataset
    - extract pose from video
    - perform various segmentations (size/number)
- Run it through the model
    - retrieve probablity scores
    - find the intervals with highest probablilty scores
- Run it through some state of the art transformer bart/t5
    - generate most probable sentence
- Evaluate generated sentence based on true video sentence (BLEU/ROGUE)

![pipeline-image](<AML Pipeline.png>)


    