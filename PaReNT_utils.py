import re
import subprocess
import uuid
import os
import sys

import pandas as pd
import numpy as np

try:
    from transcription.latin_to_cyril import latin_to_cyril
    from transcription.cyril_to_latin import cyril_to_latin
except:
    from latin_to_cyril import latin_to_cyril
    from cyril_to_latin import cyril_to_latin

from typing import List, Dict, Tuple, Any, Literal

Vector = List[float]
StringVec = List[str]
IntVec = List[int]
TupleList = List[Tuple[str, str]]

def contains_empty_lst(inp):
    return(any([ []==i for i in inp]))

def grapheme_encode(x):
    if type(x) == str:
        x = x.replace(" ", "_")
        return (" ".join([i for i in x]))
    if type(x) == list:
        lst = []
        for i in x:
            i = i.replace(" ", "_")
            lst.append(" ".join([y for y in i]))
        return (lst)

def grapheme_decode(x):
    if type(x) == str:
        return (x.replace(" ", "").replace("_", " "))
    if type(x) == list:
        lst = []
        for i in x:
            i = i.replace(" ", "").replace("_", " ")
            lst.append(i)
        return (lst)

def grapheme_decode_PaReNT(x):
    if type(x) == str:
        lang = x[-2:]
        x = x[:-2]
        return (x.replace(" ", "").replace("_", " "), lang)
    if type(x) == list:
        lst = []
        for i in x:
            lang = i[-2:]
            i = i[:-2]
            i = i.replace(" ", "").replace("_", " ")
            lst.append((i, lang))
        return (lst)

def decide_word_origin(lemmas: list, retrieved_lst: list):
    output_lst = []

    for lemma, retrieved in zip(lemmas, retrieved_lst):
        lemma, retrieved = lemma.lower(), retrieved.lower()
        if " " in retrieved:
            output_lst.append("Compound")
        elif lemma == retrieved:
            output_lst.append("Unmotivated")
        else:
            output_lst.append("Derivative")

    return output_lst

def split(lemmalist, modelnum="", n_best=1):
    odhad = []
    sample = []
    modelnum = str(modelnum)

    job_id = str(uuid.uuid1())
    input_filename = "input_marian_identify.txt" + job_id
    output_filename = "output_marian_identify.txt" + job_id

    with open("../" + input_filename, "w") as file:
        for slovo in lemmalist:
            ##FIXME tohle může dělat divnej bordel
            slovo = str(slovo)
            file.writelines(grapheme_encode(slovo) + "\n")

    if modelnum == "":
        bashCommand = "cd ../Marian_GPU/ \n ./marian/build/marian-decoder " \
                      "-m model.npz -v corpus_in.yml corpus_out.yml --beam-size {} --n-best < ../{} > ../" \
                      "{}".format(n_best, input_filename, output_filename)

    elif re.match("ensemble*", modelnum):
        modelnums = modelnum.replace("ensemble_", "").split("_")
        modelnum1 = modelnums[0]
        modelnum2 = modelnums[1]
        modelnum3 = modelnums[2]
        bashCommand = "cd ../Marian_GPU/ \n" \
                      "./marian/build/marian-decoder " \
                      "--models model.iter{}.npz model_transformer.iter{}.npz model_rnn.iter{}.npz " \
                      "--weights 0.65 0.35 0.2 " \
                      "-v corpus_in.yml corpus_out.yml " \
                      "--beam-size {} --n-best < ../{} > ../" \
                      "{}".format(modelnum1, modelnum2, modelnum3, n_best, input_filename, output_filename)


    else:
        bashCommand = "cd ../Marian_GPU/ \n" \
                      "./marian/build/marian-decoder " \
                      "-m model.iter{}.npz " \
                      "-v corpus_in.yml corpus_out.yml " \
                      "--beam-size {} --n-best < ../{} > ../" \
                      "{}".format(modelnum, n_best, input_filename, output_filename)

    process = subprocess.run(bashCommand, shell=True, encoding='UTF-8')

    os.remove("../" + input_filename)

    odhad_slova = []
    with open("../" + output_filename, "r") as file:
        for line in file.readlines():
            odhad_slova.append(grapheme_decode(line))

    os.remove("../" + output_filename)

    n_best = int(n_best)

    output = []
    for i in range(0, len(odhad_slova), n_best):
        if i + n_best <= len(odhad_slova):
            output.append(odhad_slova[i:i + n_best])

    output = [[y.split('|||')[1] for y in i] for i in output]
    return (output)

def tree_accuracy(model_output: StringVec, ground_truth: StringVec, data_source) -> float:
    assert len(model_output) == len(ground_truth)

    acc_lst = []
    for x, Y in zip(model_output, ground_truth):
        if x == Y:
            acc_lst.append(True)
        else:
            x = x.split(" ")
            Y = Y.split(" ")

            x_lexemes_lst = [data_source.get_lexemes(i) for i in x]
            Y_lexemes_lst = [data_source.get_lexemes(i) for i in Y]

            if (not (contains_empty_lst(x_lexemes_lst) or contains_empty_lst(Y_lexemes_lst))) and (len(x_lexemes_lst) == len(Y_lexemes_lst)):
                inner_lst = []

                for x_lexemes,Y_lexemes in zip(x_lexemes_lst, Y_lexemes_lst):
                    inner_lst.append(
                        any([x_lexeme.get_tree_root().lemid == Y_lexeme.get_tree_root().lemid for x_lexeme, Y_lexeme in
                             product(x_lexemes, Y_lexemes)]))

                acc_lst.append(all(inner_lst))
            else:
                acc_lst.append(False)

    return sum(acc_lst) / len(acc_lst)

def retrieval_accuracy(truth: pd.Series, pred: pd.Series):
    len_truth = len(truth)
    len_pred = len(pred)
    assert len_truth == len_pred

    correct = 0
    for a, b in zip(truth, pred):
        if a == b:
            correct += 1
    return round((correct / len_truth), 2)

def correct_type(x: str):
    try:
        return int(x)
    except:
        try:
            return float(x)
        except:
            return x

def parse_out_important(dirname, return_bs=True):
    important_parameters = ["units", "char_embedding_dim", "transblocks", "attention_units",
                            "batch_sz", "dropout", "recurrent_dropout"]
    important_parameters_dict = {i[0:3]: i for i in important_parameters}
    dict_all = {i.split("=")[0]: i.split("=")[1] for i in dirname.split("-") if "=" in i}

    out_dict = {important_parameters_dict[i[0]]: correct_type(i[1]) for i in dict_all.items() if
                i[0] in important_parameters_dict.keys()}

    if "att" not in dict_all.keys():
        out_dict["attention_units"] = out_dict["units"]

    if return_bs is False:
        out_dict.pop("batch_sz")

    return out_dict

def good_candidates(input_word: str, list_of_candidates: list, classifier_prediction: int) -> list:
    if classifier_prediction == 0:
        output = [input_word]

    elif classifier_prediction == 1:
        output = [i for i in list_of_candidates if (i != input_word and " " not in i)]

    elif classifier_prediction == 2:
        output = [i for i in list_of_candidates if (i != input_word and " " in i)]
    else:
        raise Exception("Meaningless classification prediction")

    if output == []:
        return [list_of_candidates[0]]
    else:
        return output


