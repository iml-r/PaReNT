import pandas as pd

#Accuracy of PaReNT adjusted by ignoring some types of errors
germanet_errors_sample = pd.read_csv("./scripts/PaReNT_final_evaluation/germanet/germanet_errors_sample_annotated.tsv", sep="\t")
valid_errors_lst = ["M", "3", "4", "6", "7"]
valid_errors_percentage = sum([a in valid_errors_lst for a in germanet_errors_sample.Error_type])/len(germanet_errors_sample)

germanet = pd.read_csv("./scripts/PaReNT_final_evaluation/germanet/germanet.tsv", sep="\t")
germanet_acc = sum(germanet.parents == germanet.PaReNT_retrieve)/len(germanet)
adjusted_germanet_acc = germanet_acc + (1-germanet_acc)*(1-valid_errors_percentage)
print(f"The accuracy of PaReNT adjusted for Krotova is {round(adjusted_germanet_acc, 2)}")

#Accuracy of Krotova's splitter by interpreting it as a percentage of the oracle score as determined by our criteria
#Criteria = the returned parents must precisely match the label parents
#Oracle score is calculated by determining the proportion of compounds in GermaNet which are simple concatenations of their parents
#Otherwise, a split-point approach cannot return the correct parents, because the interfix has to end up on one of the words
germanet = pd.read_csv("./scripts/PaReNT_final_evaluation/germanet/germanet.tsv", sep="\t")
#Remove hyphenated compounds because I do not know how they were handled
germanet_unhyphenated = germanet[["-" not in i for i in germanet.lexeme]]
#Check how many
germanet_hyphenated = germanet[["-" in i for i in germanet.lexeme]]

oracle_score = sum([(a.casefold() == b.casefold().replace(" ", "")) for a,b in zip(germanet_unhyphenated.lexeme, germanet_unhyphenated.parents)])/len(germanet_unhyphenated)
krotova_acc = 0.956
krotova_adjusted_acc = oracle_score*krotova_acc
print(f"The accuracy of Krotova et al. adjusted for PaReNT is {round(krotova_adjusted_acc, 2)}")
