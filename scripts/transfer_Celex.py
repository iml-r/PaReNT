import pandas as pd

##English
colnames = """1.
IdNum
2.
Head
3.
Cob
4.
MorphStatus
5.
Lang
6.
MorphCnt
7.
NVAffComp
8.
Der
9.
Comp
10.
DerComp
11.
Def
12.
Imm
13.
ImmSubCat
14.
ImmSA
15.
ImmAllo
16.
ImmSubst
17.
ImmOpac
18.
TransDer
19.
ImmInfix
20.
ImmRevers
21
FlatSA
22.
StrucLab
23.
StrucAllo
24.
StrucSubst
25.
StrucOpac
26.     
NVAffComp
27.     
Der 
28.     
Comp
29.     
DerComp
30.     
Def
31.     
Imm
32.     
ImmSubCat
33.     
ImmSA
34.     
ImmAllo
35.     
ImmSubst
36.     
ImmOpac
37.     
TransDer
38.     
ImmInfix
39.     
ImmRevers
40.     
FlatSA
41.     
StrucLab 
42.     
StrucAllo
43.     
StrucSubst
44.     
StrucOpac
"""
colnames = [i for i in colnames.split("\n") if not "." in i]

df = pd.read_csv("databases/celex2/english/eml/eml.cd", delimiter="\\", error_bad_lines=False, header=None)
df.columns = colnames[0:len(df.columns)]

df.to_csv("data_raw/English/CELEX_en.tsv", sep = "\t", index=False)

##Dutch
colnames = """1.
IdNum
2.
Head
3.
Inl
4.
MorphStatus
5.
MorphCnt
6.
DerComp
7.
Comp
8.
Def
9.
Imm
10.
ImmSubCat
11.
ImmAllo
12.
ImmSubst
13.
StrucLab
14.
StrucAllo
15.
StrucSubst
16.
Sepa
17.
DerComp 
18.     
Comp
19.     
Def
20.
Imm
21.
ImmSubCat
22.
ImmAllo
23.
ImmSubst
24.
StrucLab
25.
StrucAllo
26.
StrucSubst
27.
Sepa
"""
colnames = [i for i in colnames.split("\n") if not "." in i]

df = pd.read_csv("databases/celex2/dutch/dml/dml.cd", delimiter="\\", error_bad_lines=False, header=None)
df.columns = colnames[0:len(df.columns)]

df.to_csv("data_raw/Dutch/CELEX_nl.tsv", sep = "\t", index=False)

##German
colnames =    """1.
IdNum
2.
Head
3.
Mann
4.
MorphStatus
5.   
MorphCnt
6.   
DerComp
7.   
Comp
8.   
Def
9.
Imm
10.
ImmClass
11.
ImmAllo
12.
ImmOpac
13.
ImmUml
14.
StrucLab
15.
StrucAllo
16.
StrucOpac
17.
StrucUml
18.
Sepa
19.
InflPar
20.
InflVar
"""
colnames = [i for i in colnames.split("\n") if not "." in i]

df = pd.read_csv("databases/celex2/german/gml/gml.cd", delimiter="\\", error_bad_lines=False, header=None)
df.columns = colnames[0:len(df.columns)]

df.to_csv("data_raw/German/CELEX_de.tsv", sep = "\t", index=False)