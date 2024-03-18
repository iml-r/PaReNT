#!/usr/bin/env python3
import argparse
import sys,time,os,textwrap,shutil
import warnings

import pandas as pd
import numpy as np

from multiprocessing import Process

parser = argparse.ArgumentParser()
parser.add_argument("-i", action="store_true", help="Interactive mode. For playing around only.")
parser.add_argument("--debug_mode", action="store_true", help="Prints out various diagnostic and warning messages.")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size of the model. Larger batch size is faster, but consumes more memory. Negligible effect when inferring on CPU.")
parser.add_argument("--num_candidates", default=6, type=int, help="Number of candidate retrievals PaReNT will generate. More candidates means a higher probability of generating the correct retrieval, but consumes more resources.")
args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
languages = ["en", "de", "nl", "es", "fr", "cs", "ru"]
class_dict = {0: "Unmotivated", 1: "Derivative", 2: "Compound"}

dirname = "e17-arc=anime_body_pillow-clu=True-bat=64-epo=1000-uni=2048-att=128-cha=64-tes=0-tra=1-len=0.0-fra=1-lr=0.0001-opt=Adam-dro=0.2-rec=0.5-l=l1-use=1-neu=0-neu=0-sem=0"

if args.i is True:
    args.batch_size = 1


try:
    with open("PaReNT_ascii_logo") as f:
        logo = f.read()
except:
    with open("../PaReNT_ascii_logo") as f:
        logo = f.read()


if not args.i:
    if sys.stdin.isatty():
        columns = shutil.get_terminal_size().columns
        for line in logo.split("\n"):
            print(" "*(columns//6) + line, end="\n")

        print(
            '''
           PaReNT v. 0.7b

           This is PaReNT (Parent Retrieval Neural Tool), a deep-learning-based multilingual tool performing parent retrieval and word formation classification in English, German, Dutch, Spanish, French, Russian, and Czech. 

           Parent retrieval refers to determining the lexeme or lexemes the input lexeme was based on. Think of it as an extension of lemmatization.
           For example, `darkness' is traced back to `dark'; `waterfall' decomposes into `water' and `fall'. 

           Word formation classification refers to determining the input lexeme as a compound (e.g. `proofread'), a derivative (e.g. `deescalate') or as an unmotivated word (e.g. `dog').
           It also estimates the probability of each class.

           If you want to play around with PaReNT, use the -i flag (i.e. 'python3 PaReNT.py -i') to run the tool interactively.

           If you want to run it on actual data, input a .tsv tab-separated file (i.e. python3 PaReNT.py < my_file.tsv > output.tsv)
           It can have any number of columns, but at least one of them must be called "lemma" and contain the standard dictionary form of the words you are interested in retrieving and/or classifying.
           Additionally, there should also be a "language" column, specifying which language each of the words comes from by way of a language token. 

           List of tokens:

           English:    en 
           German:     de
           Dutch:      nl
           Spanish:    es
           French:     fr
           Russian:    ru
           Czech:      cs

           Example .tsv file:

            lemma   language
            černobílý   cs
            brainless   en
            fiancée fr
            aardwolf    nl
            Hochschule  de


           Technically, the "language" column is not strictly necessary, but it *is* strongly recommended, as the feature is untested. 
           Foregoing its usage may result in unexpected results, but PaReNT works without it, and can return meaningful results.

           If there are more columns in your input .tsv file, PaReNT will keep them unchanged. It will add the following columns:

           1) PaReNT_retrieval_best:                Best parent(s), selected from PaReNT_retrieval_candidates based on columns 4), 5) and 6).
           2) PaReNT_retrieval_greedy:              Parent(s) retrieved using greedy decoding.
           3) PaReNT_retrieval_candidates:          All candidates retrieved using beam search decoding, sorted by score.
           4) PaReNT_Compound_probability:          Estimated probability the word is a Compound.
           5) PaReNT_Derivative_probability:        Estimated probability the word is a Derivative.
           6) PaReNT_Unmotivated_probability        Estimated probability the word is Unmotivated

           On a consumer-grade processor, PaReNT should be able to process about 1 lemma per second. A progress bar will be displayed.
            '''
            # Add the option of using a "fast" variant with only greedy decoding

        )
        sys.exit()
    else:
        print("Loading...", file=sys.stderr)
        try:
            df = pd.read_csv(sys.stdin, sep="\t", header=0)
        except Exception as e:
            raise AttributeError("Input file does not seem to be a valid .tsv file. Please run 'python3 PaReNT.py' for help.")

        try:
            lemmas = [str(i) for i in list(df.lemma)]
        except Exception as e:
            raise AttributeError("Input file seems to be missing 'lemma' column. Please run 'python3 PaReNT.py' for help.")

        try:
            language = list(df["language"])
        except:
            warnings.warn("Input file seems to be missing 'language' column. PaReNT can technically do without it, but this feature has not been tested and is not recommended.")
            language = np.repeat("UNK", len(lemmas))


class HiddenPrints:
    def __enter__(self):
        if not args.debug_mode:
            self._original_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')

            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not args.debug_mode:
            sys.stderr.close()
            sys.stderr = self._original_stderr

            sys.stdout.close()
            sys.stdout = self._original_stdout


def slow_print(text,delay_time):
  for character in text:
    sys.stdout.write(character)
    sys.stdout.flush()
    time.sleep(delay_time)

def show_logo(logo, delay_time):
    print("Loading while animation plays... \n")
    columns = shutil.get_terminal_size().columns
    for line in logo.split("\n"):
        slow_print(" "*(columns//6) + line +"\n", delay_time=delay_time)

def intro():
    show_logo(logo, delay_time=0.015)
    #sys.stdout.flush()
    slow_print(
        f"\n\n Give PaReNT a word followed by its language {languages}, separated by a space -- e.g. 'waterfall en' -- and it will find its parent or parents!" +
        f"\n\n Tip: If you don't input the language, or input one not in the list, PaReNT will try to do its thing without knowing the source language." +
        f" (Be warned: This feature is untested, and may result in highly unreliable output.) " +
        f"\n Input 'Q' to exit. (Or just press Ctrl-D.) \n \n \n", delay_time=0.01)
    sys.stdout.flush()

if args.i:
    t = Process(target=intro)
    t.start()

##This suppresses printing because a lot of this stuff displays messages the user is not interested in
with HiddenPrints():
    import tensorflow as tf
    import numpy as np

    try:
        import scripts.PaReNT_core as PaReNT_core
        import scripts.PaReNT_utils as PaReNT_utils
    except:
        import PaReNT_core as PaReNT_core
        import PaReNT_utils

    try:
        for_vocab = pd.read_csv(f"./model/{dirname}/vocab.lst", header=0, na_filter=False, skip_blank_lines=False).squeeze("columns").tolist()
    except:
        for_vocab = pd.read_csv(f"../model/{dirname}/vocab.lst", header=0, na_filter=False, skip_blank_lines=False).squeeze("columns").tolist()

    if not args.i:
        batch_size = args.batch_size
    else:
        batch_size = 1

    model = PaReNT_core.PaReNT(char_vocab=for_vocab,
                               batch_sz=batch_size,
                               train_len=543056,
                               embedding_model=PaReNT_core.multibpe_model,
                               activation=tf.nn.swish,
                               **PaReNT_utils.parse_out_important(dirname, return_bs=False))
    model.optimizer = tf.keras.optimizers.legacy.Adam()
    model.init_model()

    try:
        model.load_weights(f"./model/{dirname}/model_weights.tf")
    except:
        model.load_weights(f"../model/{dirname}/model_weights.tf")

    model.compile()
##

if args.i:
    leave = False
    t.join()
    while True:
        user_input = input("Word: ").strip().split(" ")

        if len(user_input) == 1:
            if user_input[0] == "Q":
                sys.exit("PaReNT exited successfully")
            else:
                input_word = user_input[0]
                input_language = "UNK"

        elif len(user_input) == 2:
            input_word, input_language = user_input

        else:
            print("Invalid input! Please enter a maximum of 2 words, the second of which should be a language.")
            continue

        t1 = time.time()
        output = model.retrieve_and_classify([(input_language, input_word)],
                                             threshold=1,
                                             return_probs=True,
                                             try_candidates=True)
        t2 = time.time()
        retrieved_parents, classification, retrieval_probabilities, classification_probabilities, candidates = output

        print(f"{input_word} <- {retrieved_parents[0].replace(' ', ', ')}")
        print(f"Predicted class: {class_dict[classification[0]]}, "
              f"Certainty: {round(np.max(classification_probabilities[0])*100, 4)}%")
        print(f"Time taken to retrieve parents: {np.round(t2-t1, 2)} seconds.")
        print(f"Beam search candidates: {candidates[0]}")
        print(f"Best beam search candidate: {PaReNT_utils.good_candidates(input_word, candidates[0], classification[0])[0]}  \n ---------- \n \n")

else:
    list_of_lexemes = [*zip(language,lemmas)]
    output = model.retrieve_and_classify(list_of_lexemes,
                                         threshold=batch_size,
                                         return_probs=True,
                                         try_candidates=True,
                                         num_candidates=6)

    retrieved_parents, classification, retrieval_probabilities, classification_probabilities, candidates = output
    classification_probabilities_array = np.array(classification_probabilities)

    output_df = df

    output_df["PaReNT_retrieval_best"] = [PaReNT_utils.good_candidates(lemma, loc, classpred)[0] for lemma,loc,classpred in zip(lemmas,candidates, classification)]
    output_df["PaReNT_retrieval_greedy"] = retrieved_parents
    output_df["PaReNT_retrieval_candidates"] = candidates
    output_df["PaReNT_classification"] = classification
    output_df["PaReNT_Compound_probability"] = np.round(classification_probabilities_array[:,2], 5).tolist()
    output_df["PaReNT_Derivative_probability"] = np.round(classification_probabilities_array[:, 1], 5).tolist()
    output_df["PaReNT_Unmotivated_probability"] = np.round(classification_probabilities_array[:, 0], 5).tolist()

    output_df.to_csv(sys.stdout, index_label=False, sep="\t", index=False)
    print("Done", file=sys.stderr)