try:
    import scripts.PaReNT_core as PaReNT
    import scripts.PaReNT_utils
except:
    import PaReNT_core as PaReNT
    import PaReNT_utils

try:
    import scripts.derinet
    from scripts.derinet.lexicon import Lexicon
except:
    import derinet
    from derinet.lexicon import Lexicon

import pandas as pd
import tensorflow as tf


dirname = "e13-arc=FINAL5-clu=True-bat=64-epo=1000-uni=2048-att=512-cha=64-tes=0-tra=1-len=0.0-fra=1-lr=0.0001-opt=Adam-dro=0.2-rec=0.5-l=l1-use=1-neu=0-neu=0-sem=0/"
try:
    train_df = PaReNT.load_df(subset="train",
                          cluster=False)
except:
    train_df = PaReNT.load_df(subset="train",
                          cluster=True)

print(train_df)

try:
    for_vocab = pd.read_csv(f"./tf_models/{dirname}/vocab.lst", header=0, na_filter=False, skip_blank_lines=False).squeeze("columns").tolist()
except:
    for_vocab = pd.read_csv(f"../tf_models/{dirname}/vocab.lst", header=0, na_filter=False, skip_blank_lines=False).squeeze("columns").tolist()

#for_vocab2 = PaReNT.get_vocab(train_df)
testlst = [("es", "posicionar"), ("de", "Forsche"), ("nl", "omsingelen"), ("ru", "успеваемость"), ("en", "passkey"), ("cs", "Jedličková"), ("de", "Weltmeisterschaftsspiel")]

model = PaReNT.PaReNT(char_vocab=for_vocab,
                      train_len=len(train_df),
                      embedding_model=PaReNT.multibpe_model,
                      activation=tf.nn.swish,
                      **PaReNT_utils.parse_out_important(dirname))


example = train_df.sample(1)
example = PaReNT.preprocess_data(example, start=0., end=1., model=model.embedding_model)

inp, targ = example[0]
targ_seq, targ_class = targ
dec_input = model.lookup_chars(targ_seq)
dec_input = dec_input[:, :-1]  # Ignore <end> token
real_seq = model.lookup_chars(targ_seq)
real_seq = real_seq[:, 1:]  # ignore <start> token
real_seq = real_seq.to_tensor()

retrieve_probs, classify_probs = model(inp,
                                      training=True,
                                      dec_input=dec_input)

try:
    test_data = PaReNT.load_df("test", False)
except:
    test_data = PaReNT.load_df("test", True)

inp = [*zip(test_data["language"], test_data["lexeme"])]


try:
    model.load_weights(f"./tf_models/{dirname}/model_weights.tf")
except:
    model.load_weights(f"../tf_models/{dirname}/model_weights.tf")

model.compile()

# lexicon = scripts.derinet.Lexicon()
# lexicon.load("./data_raw/Czech/derinet-2-1.tsv", on_err='continue')

model.evaluate(derinet=[],
               cluster=True,
               subset="validate",
               frac_mode=False,
               threshold=64)

# #for_analysis["PaReNT_classify_local_batch"] = model.retrieve(for_analysis["lexeme"])
# for_analysis["PaReNT_classify_local"] = model.classify(retr_inp)
# classifier_dict = {0:"Unmotivated", 1:"Derivative", 2:"Compound"}
# for_analysis["PaReNT_classify_local"] = [classifier_dict[i] for i in for_analysis["PaReNT_classify_local"]]
# newnames = ["input", "ground_truth", "cluster_predictions", "local_predictions"]
# for_analysis_relevantcols = for_analysis[["lexeme", "parents", "PaReNT_retrieve", "PaReNT_retrieve_whileloop_local"]]
#
# for_analysis_relevantcols = for_analysis_relevantcols.set_axis(newnames, axis=1)
# for_analysis_relevantcols.to_csv("comparison_cluster_gpu_cluster_gpu.tsv", sep="\t")
# print(for_analysis_relevantcols)
#
# all(for_analysis_relevantcols.cluster_predictions == for_analysis_relevantcols.local_predictions)
#
# all(for_analysis.PaReNT_classify == for_analysis.PaReNT_classify_local)

# model.save_weights("blabla")
#
#
# model2 = PaReNT.PaReNT(char_vocab=for_vocab,
#                       train_len=len(train_df),
#                       embedding_model=PaReNT.multibpe_model,
#                       **parse_out_important(dirname))
#
#
# model2.load_weights("blabla")
#
# b_class = model.classify(inp[0:1800], threshold=64)
# b_retr = model.retrieve_whileloop(inp[0:1800], threshold=256)
#
# print(all([i == y for i, y in zip(a_class, b_class)]))
# print(all([i == y for i, y in zip(a_retr, b_retr)]))
#
#
# len(model.weights)
# len(model2.weights)
#
# for a,b in zip(model.weights, model2.weights):
#     print(a==b)
#
#
#
# for a,b in zip(model.weights, model2.weights):
#     print(a==b)
#
#
# # checkpoint = tf.train.Checkpoint(model)
# # checkpoint.restore(f"./tf_models/{dirname}/model_weights.tf")
#
# # try:
# #   model.load_weights(f"./tf_models/{dirname}/model_weights.tf")
# # except:
# #   model.load_weights(f"../tf_models/{dirname}/model_weights.tf")
#
# print(model.classify(testlst))
# print(model._while_retrieve(testlst))
#
# savedir = "test_saving_models/model"
# model.save_weights(savedir)
#
# del model
# model = PaReNT.PaReNT(char_vocab=for_vocab,
#                       train_len=len(train_df),
#                       embedding_model=PaReNT.multibpe_model,
#                       **parse_out_important(dirname))
# model.load_weights(savedir, skip_mismatch=False)
#
# print(model.classify(testlst))
# print(model._while_retrieve(testlst))
#
# model.save_weights(savedir)
# del model
# model = PaReNT.PaReNT(char_vocab=for_vocab,
#                       train_len=len(train_df),
#                       embedding_model=PaReNT.multibpe_model,
#                       **parse_out_important(dirname))
# model.load_weights(savedir)
#
# print(model.classify(testlst))
# print(model._while_retrieve(testlst))
#
# model.save(savedir+"h5")