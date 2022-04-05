#!/usr/bin/env python3

import subprocess
# import sentencepiece
# import pandas as pd
# import Levenshtein as l
# import copy
# import numpy as np
import argparse
import re
#from scripts.functions import grapheme_decode_PaReNT
#import Segmenter

def parse_n_splits(string):
    string = string[0]
    lst = [i.split("|||")[1] for i in string.split("\n")]
    return(lst)

def n_decide_if_comp(string):
    e = any([" " in i for i in parse_n_splits(string)])
    if e:
        return(True)
    else:
        return(False)

def grapheme_decode_PaReNT(x):
    if type(x) == str:
        lang = x[-2:]
        x = x[:-2]
        return(x.replace(" ", "").replace("_", " "), lang)
    if type(x) == list:
        lst = []
        for i in x:
            lang = i[-2:]
            i = i[:-2]
            i = i.replace(" ", "").replace("_", " ")
            lst.append((i, lang))
        return(lst)

parser = argparse.ArgumentParser(description="Model type")
parser.add_argument('--model_type', type=str, default="gpu", help="Seš na clusteru nebo doma?")
parser.add_argument('--encoding', type=str, default="grapheme", help="Kterej encoding?")
parser.add_argument('--model_number', type=int, default=70000, help="kterej model?")
parser.add_argument('slovo', type=str, help="Zadej kompozitum brooo")
parser.add_argument('--threshold', type=str, default=1, help="Zadej threshold pro Mariana")
args = parser.parse_args()

def grapheme_encode(x):
    if type(x) == str:
        x = x.replace(" ", "_")
        return(" ".join([i for i in x]))
    if type(x) == list:
        lst = []
        for i in x:
            i = i.replace(" ", "_")
            lst.append(" ".join([y for y in i]))
        return(lst)


def grapheme_decode(x):
    if type(x) == str:
        return(x.replace(" ", "").replace("_", " "))
    if type(x) == list:
        lst = []
        for i in x:
            i = i.replace(" ", "").replace("_", " ")
            lst.append(i)
        return(lst)

def n_decide_if_unmotivated(string, slovo_in=args.slovo):
    slovo_out = parse_n_splits(string)[0]
    return(slovo_out == slovo_in)


# def rozpoznej_rodice(slovo, model_type):
#     slovo = " ".join(sp.Encode(slovo, out_type=str))
#     if model_type == "cpu":
#         bashCommand = "cd /media/emil/DATA/algoritmus-iml/algoritmus-iml/Marian/ \n echo {}".format(slovo) +  \
#                       " | ./marian/build/marian-decoder -m model.npz -v corpus_in.yml corpus_out.yml --cpu-threads 1"
#     elif model_type == "gpu":
#         bashCommand = "cd /lnet/spec/tmp/algoritmus-iml/Marian_GPU/ \n echo {}".format(slovo) + \
#                       " | ./marian/build/marian-decoder -m model.npz -v corpus_in.yml corpus_out.yml"
#     else:
#         print("bad type")
#     process = subprocess.check_output(bashCommand, shell=True, encoding='UTF-8')
#     slovo = sp.Decode(process.strip().split(" "))
#     return(slovo)

def rozpoznej_rodice(slovo, model_type, model_number, threshold=args.threshold):
    if args.encoding=="sp":
        slovo = " ".join(sp.Encode(slovo, out_type=str))
    elif args.encoding=="honza":
        slovo = " ".join(honza_encode(slovo))
        print(slovo)
    elif args.encoding == "grapheme":
        slovo = grapheme_encode(slovo)
        #print(slovo)
    if model_type == "cpu":
        bashCommand = "cd ./Marian_GPU/ \n echo {}".format(slovo) +  \
                      " | ./marian/build/marian-decoder -m model.npz -v corpus_in.yml corpus_out.yml --cpu-threads 1"
    elif model_type == "gpu":
        bashCommand = "cd ./Marian_GPU/ \n echo {}".format(slovo) + \
                      " | ./marian/build/marian-decoder " \
                      "-m model.iter{}.npz " \
                      "-v corpus_in.yml corpus_out.yml " \
                      "--quiet " \
                      "--beam-size {} --n-best".format(model_number, threshold)
    else:
        print("bad type")
    process = subprocess.check_output(bashCommand, shell=True, encoding='UTF-8')
    if args.encoding == "sp":
        slovo = sp.Decode(process.strip().split(" "))
    elif args.encoding == "honza":
        slovo = honza_decode(process.strip().split(" "))
        print(slovo)
    elif args.encoding == "grapheme":
        slovo = grapheme_decode_PaReNT(process.strip())
    else:
        print("špatnej encoding")
    return(slovo)

rodice = rozpoznej_rodice(args.slovo, model_type=args.model_type, model_number=args.model_number)
# rodice = parse_n_splits(rodice)[:-1]
# jazyk = parse_n_splits(rodice)[-1]

#print(parse_n_splits(rodice))
pocet_rodicu = len(parse_n_splits(rodice)[0])-1

if n_decide_if_comp(rodice):
    typ_slova = 'Compound'
elif n_decide_if_unmotivated(rodice, grapheme_decode(args.slovo)):
    typ_slova = 'Unmotivated'
else:
    typ_slova = 'Derivative'

with open("Marian_GPU/corpus_in") as t:
    corpus = t.readlines()

corpus = [i.strip() for i in corpus]

if args.encoding == "sp":
    corpus = sp.Decode(corpus)
    slovo = sp.Decode(args.slovo)
elif args.encoding == "honza":
    corpus = honza_decode(corpus)
    slovo = honza_decode(args.slovo)
elif args.encoding == "grapheme":
    corpus = grapheme_decode_PaReNT(corpus)[0]
    slovo = grapheme_decode(args.slovo)

videno = slovo in corpus

#print(rodice)
estimate = parse_n_splits(rodice)[0][:-3]
#print(estimate)
language = parse_n_splits(rodice)[0][-3:]
print("\n",
      f'Best ancestor estimate: {args.slovo} -> {estimate.replace(" ", ", ")} \n',
      f'Best ancestor count: {len(parse_n_splits(rodice)[0].split(" "))-1} \n',
      f'Language: {language} \n',
      f'Presence in the training data: {videno}',
      "\n",
      f'Candidate list: {parse_n_splits(rodice)}')