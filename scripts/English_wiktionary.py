import xmltodict as x
import re
import numpy as np
import pandas as pd

lst = []

def lindexsplit(List, lindex):
    index = lindex
    index.sort()

    templist1 = []
    templist2 = []
    templist3 = []

    breakcounter = 0
    finalcounter = 0

    numberofbreaks = len(index)

    lastindexval = index[-1]
    finalcounttrigger = (len(List)-(lastindexval+1))

    for indexofitem,item in enumerate(List):
        nextbreakindex = index[breakcounter]

        if indexofitem < nextbreakindex:
            templist1.append(item)
        elif breakcounter < (numberofbreaks - 1):
            templist1.append(item)
            templist2.append(templist1)
            templist1 = []
            breakcounter +=1
        elif indexofitem <= lastindexval:
            templist1.append(item)
            templist2.append(templist1)
            templist1 = []
        else:
            finalcounter += 1
            templist3.append(item)
            if finalcounter == finalcounttrigger:
                templist2.append(templist3)
    return templist2

print("starting")

with open("English/enwiktionary-20210601-pages-articles.xml", "rt") as file:
    e = file.readlines()

sample = e
del e

print("data loaded")

indices = []

for i in sample:
    if "<page>" in i:
        indices.append(1)
    else:
        indices.append(0)

indices = np.array(indices)
indices = np.where(indices == 1)[0].tolist()
#indices = [0] + indices

print("indices done")

pagelist = lindexsplit(sample, indices)
del sample

# for i in range(0, len(indices)-1):
#     pagelist.append(sample[indices[i]:indices[i+1]])

print("data chunked into pages")

dictlist = pagelist

dictlist = [x.parse("".join(i)) for i in dictlist]
print("data parsed")

pattern = re.compile(r'{{compound\|en\|[a-z|]+}}')
pagelist2 = [i for i in dictlist if pattern.search(i['page']['revision']['text']['#text'].replace("\n", " "))]

print("data filtered")

df = pd.DataFrame()
kompozita = []
rodice = []

for i in pagelist2:
    kompozita.append(i['page']['title'])
    match = pattern.search(i['page']['revision']['text']['#text'])
    rodice.append(match[0].replace("{{compound|en|", "").replace("}}", ""))

df['kompozitum'] = kompozita
df['rodiče'] = rodice

print("data ready - writing...")

df.to_csv("English/wiktionary_mined_compounds.tsv", sep="\t")