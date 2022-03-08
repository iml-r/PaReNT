import pandas
import pandas as pd
import os
from derinet.lexicon import Lexicon
from scripts.functions import has_more_parents, recursive_one_parent
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
from scripts.functions import parser_CELEX
from collections import Counter
import Levenshtein as l

# for language in os.listdir("data_raw"):
#     lst = []
#     for filename in os.listdir("data_raw/" + language):
#         if "UDer" in filename:
#             lex = Lexicon()
#             try:
#                 lex.load("data_raw/" + language + "/" + filename, on_err='continue')
#             except:
#                 print(filename, "error")
#             compounds = [i.lemma for i in lex.iter_lexemes() if has_more_parents(i)]
#
#             lst.append(len(compounds))
#     print(language, ":", sum(lst))

# def UDer_reformat(dataset_path:str, lang:str, label_dict:dict):
#     lexicon = Lexicon()
#     path = f'data_raw/{lang}/{dataset_path}'
#     print(path)
#     lexicon.load(path, on_err='continue')
#
#     df = pd.DataFrame()
#
#     parent_col = []
#     lexeme_col = []
#     for lexeme in lexicon.iter_lexemes():
#         lexeme_col.append(lexeme.lemma)
#         all_parents = lexeme.all_parents
#
#         if C:
#             parent_col.append(lexeme.lemma)
#
#         elif len(all_parents) == 1:
#             parent_col.append(lexeme.all_parents[0].lemma)
#
#         else:
#             parent_col.append(" ".join([i.lemma for i in all_parents]))
#
#     df["lexeme"] = lexeme_col
#     df["parents"] = parent_col
#     df["language"] = list(np.repeat(label_dict[lang], len(lexeme_col)))
#
#     return(df)

seed = 69
np.random.seed(69)


#AUX FUNCTIONS
def count_word_types(df:type(pd.DataFrame())):
    compounds = 0
    derivatives = 0
    unmotivated = 0

    for lemma,parents in zip(df["lexeme"], df["parents"]):
        if lemma == parents:
            unmotivated +=1
        elif len(parents.split(" ")) == 1:
            derivatives +=1
        elif len(parents.split(" ")) > 1:
            compounds +=1
        else:
            raise Exception("bruh máš blbě podmínky")

    return (compounds, derivatives, unmotivated)

def recurse_breakup_GermaNet(lexeme, df, parentline=[]):
    line = df[df["lexeme"] == lexeme]
    parents = list(line["parents"])[0].split(" ")

    for parent in parents:
        if parent not in list(df["lexeme"]) or parent.capitalize() not in list(df["lexeme"]):
            parentline = parentline + [parent]
        else:
            if parent in list(df["lexeme"]):
                parentline = recurse_breakup_GermaNet(lexeme=parent, df=df, parentline=parentline)
            else:
                parentline = recurse_breakup_GermaNet(lexeme=parent.capitalize(), df=df, parentline=parentline)

    return parentline

def row_decide_wordtype(df: pd.DataFrame):
    rows = [i[1] for i in df.iterrows()]
    wordtype_col = []

    for row in rows:
        if row['lexeme'] == df['parents']:
            wordtype_col.append("Unmotivated")
        elif len(df['parents'].split(" ")) == 1:
            wordtype_col.append("Derivative")
        elif len(df['parents'].split(" ")) > 1:
            wordtype_col.append("Compound")
        else:
            raise Exception("Meaningless word type")

    df["word_type"] = wordtype_col
    return df
####


#REFORMATTING FUNCTIONS
def CELEX_reformat(dataset_path:str, lang:str, label_dict:dict):
    output_df = pd.DataFrame()
    celex_dataframe = pd.read_csv(f'data_raw/{lang}/{dataset_path}', sep="\t")

    parent_col = []
    lexeme_col = []
    type_col = []
    block_col = []

    if lang == "English":
        struc_col = 'FlatSA'
        segmentation_col = 'Imm'

    elif lang == "Dutch":
        struc_col = 'StrucLab'
        segmentation_col = 'Imm'

    elif lang == "German":
        struc_col = 'StrucLab'
        segmentation_col = 'Imm'

    else:
        raise Exception("Unknown language!")

    for line_tup in celex_dataframe.iterrows():
        global line
        line = line_tup[1]

        segmentation = line[segmentation_col]
        lemma = line["Head"]

        if type(segmentation) != str:
            continue
        elif " " in lemma:
            continue

        else:
            segmentation = segmentation.split("+")
            if len(segmentation) == 1:
                lexeme_col.append(lemma)
                parent_col.append(lemma)
                type_col.append("Unmotivated")
                block_col.append(lemma)

            else:
                struc = line[struc_col]
                if type(struc) == float:
                    continue
                else:
                    parsed_struc = parser_CELEX(struc)

                    if len(parsed_struc) == 1:
                        lexeme_col.append(lemma)
                        parent_col.append(" ".join(parsed_struc))
                        type_col.append("Derivative")
                        block_col.append(parsed_struc[-1])

                    elif len(parsed_struc) > 1:
                        lexeme_col.append(lemma)
                        parent_col.append(" ".join(parsed_struc))
                        type_col.append("Compound")
                        block_col.append(parsed_struc[-1])

                    # TODO: FIX PARSING FUNCTION - THIS WAY, YOU ARE GETTING
                    # RID OF NEOCLASSICAL COMPOUNDS!
                    elif len(parsed_struc) == 0:
                        continue
                    # TODO
                    else:
                        lexeme_col.append(lemma)
                        parent_col.append(parsed_struc[0])
                        type_col.append("Derivative")
                        block_col.append(parsed_struc[-1])

    output_df["lexeme"] = lexeme_col
    output_df["parents"] = parent_col
    output_df["language"] = list(np.repeat(label_dict[lang], len(lexeme_col)))
    output_df["word_type"] = type_col
    output_df["block"] = block_col

    return(output_df)

def derinet_reformat(dataset_path:str, lang:str, label_dict:dict):
    output_df = pd.DataFrame()
    global lexicon

    path = f'data_raw/{lang}/{dataset_path}'
    # TODO: Uncomment
    # lexicon = Lexicon()
    # lexicon.load(path, on_err='continue')

    parent_col = []
    lexeme_col = []
    type_col = []
    block_col = []

    for lexeme in lexicon.iter_lexemes():
        one_parent_tree = recursive_one_parent(lexeme)
        is_lowercase = not lexeme.lemma[0].isupper()
        unmotivated = 'unmotivated' in lexeme.misc.keys()

        if 'Loanword' in lexeme.feats.keys():
            is_not_loanword = lexeme.feats['Loanword']
        else:
            is_not_loanword = True

        if 'corpus_stats' in lexeme.misc.keys():
            is_not_hapax = lexeme.misc['corpus_stats']['absolute_count'] > 1
        else:
            is_not_hapax = False

        if 'is_compound' in lexeme.misc.keys():
            compound_flag = lexeme.misc['is_compound']
        else:
            compound_flag = False

        parentlist = [i.lemma for i in lexeme.all_parents]
        parentlist_len = len(parentlist)
        one_parent_immediate = parentlist_len == 1
        two_parents_immediate = parentlist_len > 1

        if all([one_parent_tree, is_lowercase, unmotivated, is_not_loanword, is_not_hapax, not compound_flag]):
            parent_col.append(lexeme.lemma)
            lexeme_col.append(lexeme.lemma)
            type_col.append("Unmotivated")
            block_col.append(lexeme.get_tree_root().lemid)

        elif all([one_parent_immediate, is_not_loanword, is_not_hapax, not compound_flag]):
            lexeme_col.append(lexeme.lemma)
            parent_col.append(" ".join(parentlist))
            type_col.append("Derivative")
            block_col.append(lexeme.get_tree_root().lemid)

        elif two_parents_immediate:
            lexeme_col.append(lexeme.lemma)
            parent_col.append(" ".join(parentlist))
            type_col.append("Compound")
            block_col.append(lexeme.get_tree_root().lemid)

        else:
            continue

    output_df["lexeme"] = lexeme_col
    output_df["parents"] = parent_col
    output_df["language"] = list(np.repeat(label_dict[lang], len(lexeme_col)))
    output_df["word_type"] = type_col
    output_df['block'] = block_col

    return output_df

def UDer_reformat(dataset_path:str, lang:str, label_dict:dict):
    output_df = pd.DataFrame()
    # global lexicon

    path = f'data_raw/{lang}/{dataset_path}'
    # TODO: Uncomment
    lexicon = Lexicon()
    lexicon.load(path, on_err='continue')

    parent_col = []
    lexeme_col = []
    type_col = []
    block_col = []

    for lexeme in lexicon.iter_lexemes():
        parents = lexeme.all_parents
        len_parent = len(parents)


        if len_parent == 0:
            if len(lexeme.lemma) < 7:
                parent_col.append(lexeme.lemma)
                lexeme_col.append(lexeme.lemma)
                type_col.append("Unmotivated")
                block_col.append(lexeme.get_tree_root().lemid)

            else:
                continue

        elif len_parent == 1:
            lexeme_col.append(lexeme.lemma)
            parent_col.append(" ".join([i.lemma for i in parents]))
            type_col.append("Derivative")
            block_col.append(lexeme.get_tree_root().lemid)

        else:
            lexeme_col.append(lexeme.lemma)
            parent_col.append(" ".join([i.lemma for i in parents]))
            type_col.append("Compound")
            block_col.append(lexeme.get_tree_root().lemid)


    output_df["lexeme"] = lexeme_col
    output_df["parents"] = parent_col
    output_df["language"] = list(np.repeat(label_dict[lang], len(lexeme_col)))
    output_df["word_type"] = type_col
    output_df["block"] = block_col

    return(output_df)

def GoldenCompounds_reformat(dataset_path:str, lang:str, label_dict:dict):
    path = f'data_raw/{lang}/{dataset_path}'
    input_df = pd.read_csv(path, sep="\t")
    output_df = pd.DataFrame()

    logical = []
    for line in input_df.iterrows():
        line = line[1]

        if (not pd.isnull(line["word_d"]) and not pd.isnull(line["words_m"]) and not pd.isnull(line["word_h"])):
            logical.append(True)
        else:
            logical.append(False)

    input_df = input_df[logical]

    parent_col = [" ".join([y.replace(";", " "),i]) for y,i in zip(list(input_df["words_m"]), list(input_df["word_h"]))]

    output_df["lexeme"] = input_df["word_d"]
    output_df["parents"] = parent_col
    output_df["language"] = list(np.repeat(label_dict[lang], len(parent_col)))
    output_df["word_type"] = list(np.repeat("Compound", len(parent_col)))

    block_col  = []
    for line in output_df.iterrows():
        line = line[1]
        block_col.append(line["parents"].split(" ")[-1])

    output_df["block"] = block_col

    return output_df

def wiktionary_reformat(dataset_path:str, lang:str, label_dict:dict):
    path = f'data_raw/{lang}/{dataset_path}'
    input_df = pd.read_csv(path, sep="\t", index_col=0)
    input_df.columns = ["lexeme", "parents"]
    input_df["language"] = list(np.repeat(label_dict[lang], len(input_df)))
    input_df["word_type"] = list(np.repeat("Compound", len(input_df)))
    input_df["block"] = [i.split(" ")[-1] for i in input_df["parents"]]

    input_df = input_df.dropna()
    output_df = input_df[[" " not in i for i in input_df["lexeme"]]]
    return output_df

def GermaNet_reformat(dataset_path:str, lang:str, label_dict):
    path = f'data_raw/{lang}/{dataset_path}'
    input_df = pd.read_csv(path, sep="\t")
    output_df = pd.DataFrame()

    input_df.columns = ["lexeme", "parents", "block"]
    input_df = input_df.dropna()
    input_df = input_df.drop(0, axis=0)
    input_df['parents'] = [str(i).split("|")[0] for i in input_df["parents"]]
    input_df["parents"] = [i + " " + y for i,y in zip(input_df["parents"], input_df["block"])]


    output_df["lexeme"] = input_df["lexeme"]

    # parent_col = []
    # for lexeme in tqdm(input_df["lexeme"]):
    #     parent_col.append(" ".join(recurse_breakup_GermaNet(lexeme, input_df)))
    #
    # print(parent_col)
    # output_df["parents"] = parent_col
    output_df["parents"] = input_df["parents"]
    output_df["language"] = list(np.repeat(label_dict[lang], len(output_df)))
    output_df["word_type"] = list(np.repeat("Compound", len(output_df)))
    output_df["block"] = input_df["block"]

    return output_df

####


label_dict = {"Czech": "cs",
              "Dutch": "nl",
              "English": "en",
              "French": "fr",
              "German": "de",
              "Russian": "ru",
              "Spanish": "es"}

df_dict = {}

for lang in tqdm(os.listdir("data_raw")):
    df_lst = []
    for dataset_name in os.listdir("data_raw/" + lang):
        if "CELEX" in dataset_name and dataset_name[0] != ".":
            df = CELEX_reformat(dataset_path=dataset_name,
                                        lang=lang,
                                        label_dict=label_dict)
            df.name = dataset_name
            df_lst.append(df)

        elif dataset_name == "derinet-2-1.tsv":
            df = derinet_reformat(dataset_path=dataset_name,
                                        lang=lang,
                                        label_dict=label_dict)
            df.name = dataset_name
            df_lst.append(df)

        elif "UDer" in dataset_name:
            df = UDer_reformat(dataset_path=dataset_name,
                                  lang=lang,
                                  label_dict=label_dict)
            df.name = dataset_name
            df_lst.append(df)

        elif dataset_name == "GoldenCompounds.tsv":
            df = GoldenCompounds_reformat(dataset_path=dataset_name,
                                        lang=lang,
                                        label_dict=label_dict)
            df.name = dataset_name
            df_lst.append(df)

        elif dataset_name == "wiktionary_mined_compounds.tsv":
            df = wiktionary_reformat(dataset_path=dataset_name,
                                          lang=lang,
                                          label_dict=label_dict)
            df.name = dataset_name
            df_lst.append(df)

        elif "GermaNet" in dataset_name:
            df = GermaNet_reformat(dataset_path=dataset_name,
                                     lang=lang,
                                     label_dict=label_dict)
            df.name = dataset_name
            df_lst.append(df)

    df_dict[lang] = df_lst

#df = UDer_reformat("UDer-1.1-es-DeriNetES.tsv", "Spanish", label_dict)

#output log which raw databases were used in the creation of the dataset, and how many compounds/whatever
#were there
table = PrettyTable()
table.field_names = ["Data origin", "language", "Compound no.", "Derivative no.", "Unmotivated no."]
sum_comps = 0
sum_ders = 0
sum_unmotivated = 0
check_structure = []

for lang in df_dict.keys():
    for data_source in df_dict[lang]:
        counts = count_word_types(data_source)
        sum_comps += counts[0]
        sum_ders += counts[1]
        sum_unmotivated += counts[2]
        table.add_row([data_source.name,
                       lang,
                       format(counts[0], ","),
                       format(counts[1], ","),
                       format(counts[2], ",")])

        if str(data_source.columns) == "Index(['lexeme', 'parents', 'language', 'word_type'], dtype='object')":
            print(data_source.name)

        check_structure.append(data_source.columns)

table.add_row(["All sources",
               "All languages",
               format(sum_comps, ","),
               format(sum_ders, ","),
               format(sum_unmotivated, ",")])

print(table, file=open("data/data_log.txt", "w"))
print("Data reformatted! \n RUNDOWN: \n")
print(table)
print(f"Data structure unified: {(len(set([str(i) for i in check_structure]))) == 1}")

###Check no. of lexical blocks (families)
for lang in df_dict.keys():
    for data_source in df_dict[lang]:
        print(data_source.name, len(data_source)/len(set(list(data_source["block"]))))
        print("Most common block:", Counter(list(data_source["block"])).most_common()[0])

def train_test_validate_split(dataframe:type(pd.DataFrame()), lang:str=lang, seed:int=seed):
    blocks = list(set(list(dataframe["block"])))
    validate_blocks_size = round(200 / (len(data_source)/len(blocks)))
    df_name = dataframe.name
    np.random.seed(seed)

    validate_blocks = np.random.choice(blocks,size=validate_blocks_size,replace=False)
    for block in validate_blocks:
        blocks.remove(block)
    validate_df = dataframe[[i in validate_blocks for i in dataframe["block"]]]

    test_blocks = np.random.choice(blocks,size=round(0.2*len(blocks)),replace=False)
    for block in test_blocks:
        blocks.remove(block)
    test_df = dataframe[[i in test_blocks for i in dataframe["block"]]]

    train_df = dataframe[[i in blocks for i in dataframe["block"]]]

    # blocks2 = list(set(list(dataframe["block"])))
    #
    # validate_check = list(validate_df["block"])
    # test_check = list(test_df["block"])
    # train_check = list(train_df["block"])
    # for block in tqdm(blocks2):
    #     validation_presence = block in validate_check
    #     test_presence = block in test_check
    #     train_presence = block in train_check
    #
    #     if sum([validation_presence, test_presence, train_presence]) > 1:
    #         raise Exception("Dataset contamination!")

    validate_df.to_csv(f"data/validate/{lang}_{df_name}-validate.csv", sep=".")
    test_df.to_csv(f"data/test/{lang}_{df_name}-test.csv", sep=".")
    train_df.to_csv(f"data/train/{lang}_{df_name}-train.csv", sep=".")


for lang in tqdm(df_dict.keys()):
    for data_source in df_dict[lang]:
        train_test_validate_split(data_source, lang=lang, seed=seed)

print("Done!")