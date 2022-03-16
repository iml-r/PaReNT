import pandas as pd
import os
from scripts.functions import grapheme_decode_PaReNT,grapheme_decode,grapheme_encode
import re
import subprocess

def split(lemmalist, modelnum="", n_best=1):
    odhad = []
    sample = []

    with open("input_marian_identify.txt", "w") as file:
        for slovo in lemmalist:
            file.writelines(grapheme_encode(slovo) + "\n")

    if modelnum == "":
        bashCommand = "cd ./Marian_GPU/ \n ./marian/build/marian-decoder " \
                  "-m model.npz -v corpus_in.yml corpus_out.yml --beam-size {} --n-best < ../input_marian_identify.txt > ../" \
                  "output_marian_identify.txt".format(n_best)

    elif re.match("ensemble*", modelnum):
        modelnums = modelnum.replace("ensemble_", "").split("_")
        modelnum1 = modelnums[0]
        modelnum2 = modelnums[1]
        modelnum3 = modelnums[2]
        bashCommand = "cd ./Marian_GPU/ \n" \
                      "./marian/build/marian-decoder " \
                      "--models model.iter{}.npz model_transformer.iter{}.npz model_rnn.iter{}.npz " \
                      "--weights 0.65 0.35 0.2 " \
                      "-v corpus_in.yml corpus_out.yml " \
                      "--beam-size {} --n-best < ../input_marian_identify.txt > ../" \
                      "output_marian_identify.txt".format(modelnum1, modelnum2, modelnum3, n_best)


    else:
        bashCommand = "cd ./Marian_GPU/ \n" \
                      "./marian/build/marian-decoder " \
                      "-m model.iter{}.npz " \
                      "-v corpus_in.yml corpus_out.yml " \
                      "--quiet-translation --quiet " \
                      "--beam-size {} --n-best < ../input_marian_identify.txt > ../" \
                      "output_marian_identify.txt".format(modelnum, n_best)

    process = subprocess.run(bashCommand, shell=True, encoding='UTF-8')

    odhad_slova = []

    with open("output_marian_identify.txt", "r") as file:
        for line in file.readlines():
            odhad_slova.append(grapheme_decode(line))

    n_best = int(n_best)

    output = []
    for i in range(0, len(odhad_slova), n_best):
        if i + n_best <= len(odhad_slova):
            output.append(odhad_slova[i:i + n_best])

    output = [[y.split('|||')[1] for y in i] for i in output]
    return(output)

def decide_word_origin(lemmas:list, retrieved_lst:list):
    output_lst = []

    for lemma, retrieved in zip(lemmas, retrieved_lst):
        if " " in lemma:
            output_lst.append("Phrase")
        elif " " in retrieved:
            output_lst.append("Compound")
        elif lemma == retrieved:
            output_lst.append("Unmotivated")




df_lst = []
for filename in os.listdir("to-annotate"):
    df = pd.read_csv("to-annotate" + "/" + filename, sep="\t", header=None)
    df.name = filename.split(".")[0]
    df_lst.append(df)

output_df_lst = []
for df in df_lst:
    output_df = pd.DataFrame()
    output_df[0] = df[0]
    PaReNT_output  = [i[0] for i in grapheme_decode_PaReNT(split(df[0]))]
    output_df[1] = decide_word_origin(list(output_df[0]), PaReNT_output)
    output_df[2] = PaReNT_output

    output_df.to_csv(df.name + "_annotated.tsv", sep="\t")