import pandas as pd
import prettytable as pt
import numpy as np
import derinet.lexicon
from typing import List, Dict, Tuple, Any, Literal
import re
from itertools import product

Vector = List[float]
StringVec = List[str]
IntVec = List[int]
TupleList = List[Tuple[str, str]]

from collections import Counter

from sklearn.metrics import balanced_accuracy_score, accuracy_score
from scripts.PaReNT_utils import retrieval_accuracy


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def contains_empty_lst(inp):
    return(any([ []==i for i in inp]))


def tree_accuracy(model_output: StringVec, ground_truth: StringVec, data_source: derinet.lexicon.Lexicon) -> float:
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


def evaluate(data: pd.DataFrame,
             derinet,
             path):

    gb = data.groupby("language")
    dfs_by_language = [gb.get_group(group) for group in gb.groups.keys()]

    print(2)

    table = pt.PrettyTable()
    table.field_names = ["Language", "Retrieval accuracy", "Classification accuracy",
                         "Balanced classification accuracy"]

    classify_lst = []
    nonbal_classify_lst = []
    retrieve_lst = []
    print(3)

    for df_lang in dfs_by_language:
        classification_acc = balanced_accuracy_score(df_lang["word_type"], df_lang["PaReNT_classify"])
        classification_acc_nonbal = accuracy_score(df_lang["word_type"], df_lang["PaReNT_classify"])
        retrieval_acc = retrieval_accuracy(df_lang["parents"], df_lang["PaReNT_retrieve"])
        print(retrieval_acc)

        classify_lst.append(classification_acc)
        nonbal_classify_lst.append(classification_acc_nonbal)
        retrieve_lst.append(retrieval_acc)

        table.add_row([list(df_lang.language)[0], retrieval_acc, round(classification_acc_nonbal, 2),
                       round(classification_acc, 2)])

    table.add_row(
        ["Total", round(np.mean(retrieve_lst), 2), round(np.mean(nonbal_classify_lst), 2),
         round(np.mean(classify_lst), 2)])

    with open(path + "table.txt", "w+") as file:
        file.write(table.get_string())

    df = data[data.language == "cs"]
    tree_acc = tree_accuracy(list(df.PaReNT_retrieve), list(df.parents), derinet)

    with open(path + "tree_acc.txt", "w") as file:
        file.write(str(tree_acc))

    print("Done!")
    print(table)

dlex = derinet.Lexicon()
dlex.load("data_raw/Czech/derinet-2-1.tsv", on_err="continue")

eval_dataset = pd.read_csv("./scripts/PaReNT_final_evaluation/for_analysis.tsv", sep="\t")

###WARNING: perform manual chatgpt feeding and annotation here

# with open("./ChatGPT/prompt_template.txt", "r") as file:
#     template = file.read()
#
# for chunk in divide_chunks([*zip(eval_dataset.lexeme, eval_dataset.language)], 100):
#     with open("./ChatGPT/prompts.txt", "a") as file:
#         file.write(template)
#         for lexeme, lang_code in chunk:
#             file.write(lexeme + ":{" + lang_code + "}" + "\n")
#


chatgpt_output1 = pd.read_csv("./ChatGPT/ChatGPToutput.tsv", sep="\t")
chatgpt_output1.columns = ['lexeme', 'PaReNT_retrieve', 'PaReNT_classify']
for index, row in chatgpt_output1.iterrows():
    if row["PaReNT_retrieve"] == "-":
        chatgpt_output1.PaReNT_retrieve.iloc[index] = row.lexeme

    chatgpt_output1.PaReNT_retrieve.iloc[index] = row["PaReNT_retrieve"].replace(", ", " ")

chatgpt_output2 = pd.read_csv("./ChatGPT/ChatGPToutput2.tsv", sep="\t")
chatgpt_output2 = chatgpt_output2[chatgpt_output2.columns[0]].str.split(" +",expand = True)
chatgpt_output2.columns = ['lexeme', "lang_code", 'PaReNT_retrieve', 'PaReNT_classify']
chatgpt_output2 = chatgpt_output2.drop("lang_code", axis=1)
for index, row in chatgpt_output2.iterrows():
    chatgpt_output2.PaReNT_classify.iloc[index] = row["PaReNT_classify"].capitalize()

chatgpt_output = pd.concat([chatgpt_output1, chatgpt_output2], ignore_index=True)
eval_chatgpt = eval_dataset.head(300)
eval_chatgpt.PaReNT_retrieve = [*chatgpt_output["PaReNT_retrieve"]]
eval_chatgpt.PaReNT_classify = [*chatgpt_output["PaReNT_classify"]]

# ###WARNING: perform manual chatgpt feeding and annotation here

evaluate(derinet=dlex,
         path="./ChatGPT/ChatGPT_eval",
         data=eval_chatgpt)

eval_dummy = eval_dataset
dummy_model = {}

for lang in set(eval_dummy.language):
    df = eval_dummy[eval_dummy.language == lang]
    most_common_wordtype = Counter(df.word_type).most_common()[0][0]
    dummy_model[lang] = most_common_wordtype

eval_dummy.PaReNT_retrieve = eval_dummy.lexeme
eval_dummy.PaReNT_classify = [dummy_model[lang] for lang in eval_dummy.language]

evaluate(derinet=dlex,
         path="./ChatGPT/dummy_eval",
         data=eval_dummy)

evaluate(eval_dataset,
         path="./ChatGPT/PaReNT_eval",
         derinet=dlex)

# chatgpt_df = pd.read_csv("ChatGPT/functional70.tsv", sep="\t")
# parent_df = pd.read_csv("tf_models/e17-arc=Aninka_small_attention-clu=True-bat=64-epo=1000-uni="
#                         "1024-att=128-cha=64-tes=0-tra=2-len=0.0-fra=1-lr=0.0001-opt=Adam-dro=0."
#                         "2-rec=0.3-l=l1-use=1-neu=0-neu=0-sem=0/for_analysis.tsv",


