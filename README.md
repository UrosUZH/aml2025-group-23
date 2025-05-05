# aml_group


# Project scope

- This is an advanced machine learning project
- it is an extension of https://github.com/J22Melody/fairseq/tree/main/examples/MMPT
- Goal is to generate sentences from ASL videos

# Pipeline draft

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




    