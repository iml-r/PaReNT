import re
from derinet.lexeme import Lexeme

def recursive_one_parent(x):
    if 'is_compound' in x.misc:
        if x.misc['is_compound']:
            return False
        else:
            return(recursive_one_parent(x))
    else:
        parents = x.all_parents
        if len(parents) == 1:
            if 'is_compound' in parents[0].misc:
                if parents[0].misc['is_compound']:
                    return(False)
                else:
                    return(True)
            else:
                return (recursive_one_parent(parents[0]))
        if len(parents) > 1:
            return(False)
        elif len(parents) == 0:
             return(True)
        else:
            return(recursive_one_parent(parents[0]))

def has_more_parents(x):
    parents = x.all_parents

    if len(parents) > 1:
        return (True)
    else:
        return (False)

def recursive_one_parent(x):
    if 'is_compound' in x.misc:
        if x.misc['is_compound']:
            return False
        else:
            return(recursive_one_parent(x))
    else:
        parents = x.all_parents
        if len(parents) == 1:
            if 'is_compound' in parents[0].misc:
                if parents[0].misc['is_compound']:
                    return(False)
                else:
                    return(True)
            else:
                return (recursive_one_parent(parents[0]))
        if len(parents) > 1:
            return(False)
        elif len(parents) == 0:
             return(True)
        else:
            return(recursive_one_parent(parents[0]))

def is_root(x):
    if (recursive_one_parent(x)
    #and 'unmotivated' in x.misc.keys() and
    #x.feats['Loanword'] == 'False' and x.misc['corpus_stats']['absolute_count'] > 1
    ):
        return True
    else:
        return False

def count_uppercase(x: list):
    counter = 0

    for i in x:
        if not i.islower():
            counter +=1

    return counter > 1

def CELEX_categorize_lexeme(lexeme: Lexeme):
    if 'morpheme_order' not in lexeme.misc:
        return('unknown')
    else:
        morpheme_order = lexeme.misc["morpheme_order"].split(";")

        if len(morpheme_order) == 1:
            return("Unmotivated")

        else:

            if not count_uppercase(morpheme_order):
                return("Derivative")

            else:
                len_all_parents = len(lexeme.all_parents)
                if len_all_parents == 0 or len_all_parents > 1:
                    return("Compound")

                else:

                    if CELEX_categorize_lexeme(lexeme.parent) == "Compound":
                        return("Derivative")
                    else:
                        return("Compound")



def parser_CELEX(lexeme_struct: str):
    lexeme_struct = lexeme_struct.replace("(", "").replace(")", "")
    lexeme_struct = lexeme_struct.split(",")

    output = []
    for i in lexeme_struct:
        if "|" not in i:
            output.append(i.split("[")[0])


    return output

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