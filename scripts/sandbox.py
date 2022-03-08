#!/usr/bin/env python3

import pandas as pd

import derinet.lexeme
from derinet.lexicon import Lexicon
from scripts.functions import recursive_one_parent, has_more_parents

lexicon = Lexicon()
lexicon.load("data_raw/Czech/derinet-2-1.tsv", on_err='continue')




compounds_from_derinet = [i.lemma for i in lexicon.iter_lexemes() if has_more_parents(i)]
compounds_rev = pd.read_csv("data_raw/Czech/COMPOUND_train_annotated_REV.csv")


##DATA EXPLORATION TO FIGURE OUT HOW TO SORT OUT COMPOUNDHOOD##
from collections import Counter

lexicon = Lexicon()
lexicon.load("data_raw/German/UDer-1.0-de-GCelex.tsv")


lst = []

for i in lexicon.iter_lexemes():
    if 'morpheme_order' in i.misc:
        lst.append(i.misc['morpheme_order'])


print(Counter(lst))

lst = []
for i in lexicon.iter_lexemes():
    if 'morpheme_order' in i.misc:
        if i.misc['morpheme_order'] == "N":
            lst.append(i.lemma)

print(lst)

def count_uppercase(x: lst):
    counter = 0

    for i in x:
        if not i.islower():
            counter +=1

    return counter > 1

# def CELEX_categorize_lexeme(lexeme: derinet.lexeme.Lexeme):
#     if 'morpheme_order' not in lexeme.misc:
#         return('unknown')
#     else:
#         morpheme_order = lexeme.misc["morpheme_order"].split(";")
#
#         if len(morpheme_order) == 1:
#             return("Unmotivated")
#
#         else:
#
#             if not count_uppercase(morpheme_order):
#                 return("Derivative")
#
#             else:
#                 len_all_parents = len(lexeme.all_parents)
#                 if len_all_parents == 0 or len_all_parents > 1:
#                     return("Compound")
#
#                 else:
#
#                     if CELEX_categorize_lexeme(lexeme.parent) == "Compound":
#                         return("Derivative")
#                     else:
#                         return("Compound")


dataframe = pd.read_csv("data_raw/English/CELEX_en.tsv", sep="\t")






#[i.lemma for i in lexicon.iter_lexemes() if CELEX_categorize_lexeme(i) == "Derivative"]