import pandas as pd
from scripts.functions import grapheme_encode
import os

def grapheme_encode_PaReNT(string:str, lang_code:str):
    return(grapheme_encode(string+" ") + " " + lang_code)

df_list = []
for data_source in os.listdir("data/train"):
    df_list.append(pd.read_csv("data/train/" + data_source, sep="."))

all_data_train = pd.concat(df_list)

all_data_train = all_data_train.dropna()

input_lst = []
output_lst = []
for row in all_data_train.iterrows():
    row = row[1]
    input_lst.append(grapheme_encode(row["lexeme"]))
    output_lst.append(grapheme_encode_PaReNT(row["parents"], row["language"]))

print(len(input_lst) == len(output_lst))

with open("Marian_GPU/corpus_in", "w") as file:
    for i in input_lst:
        file.write(i + "\n")

with open("Marian_GPU/corpus_out", "w") as file:
    for i in output_lst:
        file.write(i + "\n")