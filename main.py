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
from pathlib import Path
from src.Transformer import TransformerDecoder
import src.Extraction as Extraction
import src.Evaluation as Evaluation
import sliding_window as sliding_window
import generate_poses as generate_poses
from src.Evaluation import evaluate, make_sentences, save_to_csv
def run_mock_evaluation():
    make_sentences()
    results, header = evaluate(
    reference_path="mock_sentences/reference.txt",
    hypothesis_paths=[
        "mock_sentences/reference.txt",
        "mock_sentences/shuffled.txt",
        "mock_sentences/rand_03.txt",
        "mock_sentences/rand_05.txt",
        "mock_sentences/rand_09.txt",
       
    ],
    labels=["Reference", "Shuffled", "Noisy 30%", "Noisy 50%", "Noisy 90%"],
    use_bert=False,
    use_cosine=True
)
    save_to_csv(results, header, output_path="mock_sentences/evaluation_results.csv")
  
import pandas as pd
def load_top_canditates(output_csv_path, k=10):

    df = pd.read_csv(output_csv_path)
    all_candidate_lists = []

    for _, row in df.iterrows():
        candidate_list = []
        for i in range(1, k+1):  # label_1 to label_10
            label = str(row[f'label_{i}']).lower()
            score = float(row[f'score_{i}'])
            candidate_list.append((label, score))
        all_candidate_lists.append(candidate_list)
    
    return all_candidate_lists

def run():
    pass

from sliding_window import embed_and_score_pose_segments_pipe
import utils.utils as utils
import sys
if __name__ == '__main__':
    mock_dir = Path("aml/mock_videos")
    mock_dir = Path("aml/test_folder")
    
    results_dir = Path("aml/signclip_results")
    # results_dir = Path("aml/signclip_results_vocab_new")
    results_dir.mkdir(exist_ok=True)
    
    # DO VOCAB EXTRACTION TODO ROHAM EMBEDS
    text_embeddings, vocab_words = Extraction.get_vocab_embeddings()
    print(f"Loaded {len(text_embeddings)} text embeddings and {len(vocab_words)} vocab words.")
    
    # CREATE POSE FROM VIDEOS OF DIRECTORY

    generate_poses.generate_pose_files(mock_dir)
    pose_files = list(mock_dir.glob("*.pose"))
    vocab_embed_dir = Path("aml/src/gloss_vocab_embed")
    vocab_text_path = Path("aml/src/gloss_vocab.txt")
    # vocab_embed_dir = Path("aml/src/vocab_embed_new")
    # vocab_text_path = Path("aml/src/vocab_new.txt")
    
    beam_sizes = [1] 
    beam_size = beam_sizes[0]
    
    decoder = TransformerDecoder(
                        lm_name="microsoft/phi-4-mini-instruct",
                        beam_size=beam_size,
                        alpha=0.95,
                        beta=0.05,
                        device="cuda",
                        load_in_8bit=False,
                        refine=False,
                        debug=False,
                    )
    for pose_path in pose_files:
        print(f"🔍 Processing: {pose_path.name}")
       
       
    # SEGMENT POSES, EMBED AND SCORE 
        window_sizes = [11]  
        stride_sizes = [11]   
        model_name = "asl_finetune"
        top_k = 100
        for window_size, stride in zip(window_sizes, stride_sizes):
           
            csv_filename = f"{pose_path.stem}_W{window_size}_S{stride}.csv"
            output_csv_path = results_dir / csv_filename
            embed_and_score_pose_segments_pipe(
                pose_path=pose_path,
                vocab_embed_path=vocab_embed_dir,
                vocab_text_path=vocab_text_path,
                output_csv_path=output_csv_path,
                window_size=window_size,
                stride=stride,
                model_name=model_name,
                k=top_k
            )
                
                
            compare_csv_path= Path("aml/results/sentences_compare.csv")
            alignment_csv_path = Path("aml/mock_sentences/how2sign_realigned_val.csv")
            save_path = Path("aml/results/sentences_compare.csv")
             
            sentence_name = pose_path.name.replace(".pose", "")
            k = 5
            candidate_lists, df = utils.load_top_candidates(output_csv_path, k=k)
            do_mock = False
            for beam_size in beam_sizes:
                if utils.should_skip_decoding(compare_csv_path, output_csv_path, sentence_name, beam_size, top_k=k):
                    print("⚠️ Skipping decoding — result already exists.")
                    sentence = None
            # Transformer model gets csv
                elif do_mock:
                    beam_size = 0
                    sentence = "this is a mock sentence"
                    utils.create_sentence_comparison_csv(
                    output_csv_path, 
                    sentence_name, 
                    sentence, 
                    alignment_csv_path, 
                    save_path,
                    beam_size=beam_size,
                    k = k
                )
                else:
                    # Run decoder only if not skipped
                   
                    decoder.scoring_mode = "default"
                    sentence = decoder.decode(candidate_lists)
                    # sentence = sentence.replace('"', '""')
                    utils.create_sentence_comparison_csv(
                    output_csv_path, 
                    sentence_name, 
                    sentence, 
                    alignment_csv_path, 
                    save_path,
                    beam_size=beam_size,
                    k= k
                )
        
        
        
    # window_size = 32
    # stride = 16
    # model_name = "asl_finetune"
    # top_k = 10
    # for pose_path in pose_files:
    #     print(f"🔍 Processing: {pose_path.name}")
        
       
    # pose_path = pose_files[0]  
    # csv_filename = f"{pose_path.stem}_W{window_size}_S{stride}.csv"
    # output_csv_path = results_dir / csv_filename

    # # Run embedding + scoring pipeline
    # embed_and_score_pose_segments_pipe(
    #     pose_path=pose_path,
    #     vocab_embed_path=vocab_embed_dir,
    #     vocab_text_path=vocab_text_path,
    #     output_csv_path=output_csv_path,
    #     window_size=window_size,
    #     stride=stride,
    #     model_name=model_name,
    #     k=top_k
    # )
    
           
            
            
    ######### EVAL
    utils.augment_with_window_stride(
    input_csv_path="aml/results/sentences_compare.csv",
    output_csv_path="aml/results/sentences_compare_augmented.csv"
    )
    # input_path = Path("aml/results/sentences_compare.csv")
    input_path = Path("aml/results/sentences_compare_augmented.csv")
    output_path = Path("aml/results/sentences_compare_evaluated.csv")
    utils.evaluate_sentence_csv_rows(input_path, output_path)
    utils.table_summary()     
    
            
            
            
            
    
    # scores prep with softmax
    # structure same as in esample canditates
    # default run
    #   get sentence save to file with same name
    #
    # run mock evaluation
    #   get filename go to csv and get sentence only first
    #
    
    
  
    # run_mock_evaluation()
    # run()