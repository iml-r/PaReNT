try:
    import scripts.PaReNT_tensorflow as PaReNT
except:
    import PaReNT_tensorflow as PaReNT

import pandas as pd
import tensorflow as tf

def correct_type(x: str):
    try:
        return int(x)
    except:
        try:
            return float(x)
        except:
            return x

def parse_out_important(dirname):
    important_parameters = ["units", "char_embedding_dim", "transblocks", "attention_units",
                            "batch_sz", "dropout", "recurrent_dropout"]
    important_parameters_dict = {i[0:3]: i for i in important_parameters}
    dict_all = {i.split("=")[0]: i.split("=")[1] for i in dirname.split("-") if "=" in i}

    out_dict = {important_parameters_dict[i[0]]: correct_type(i[1]) for i in dict_all.items() if
                i[0] in important_parameters_dict.keys()}

    if "att" not in dict_all.keys():
        out_dict["attention_units"] = out_dict["units"]

    return out_dict

dirname = "e19-arc=Aninka_fixed_cudnn-clu=True-bat=64-epo=1000-uni=2048-att=512-cha=64-tes=0-tra=1-len=0.0-fra=1-lr=0.0001-opt=Adam-dro=0.3-rec=0.5-l=l1-use=1-neu=0-neu=0-sem=0/"
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
                      **parse_out_important(dirname))

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

model2 = PaReNT.PaReNT(char_vocab=for_vocab,
                      train_len=len(train_df),
                      embedding_model=PaReNT.multibpe_model,
                      **parse_out_important(dirname))

retrieve_probs, classify_probs = model2(inp,
                                      training=True,
                                      dec_input=dec_input)

len(model.weights)
len(model2.weights)

for a,b in zip(model.weights, model2.weights):
    print(a==b)


model.save_weights("blabla")
model2.load_weights("blabla")
for a,b in zip(model.weights, model2.weights):
    print(a==b)


# checkpoint = tf.train.Checkpoint(model)
# checkpoint.restore(f"./tf_models/{dirname}/model_weights.tf")

# try:
#   model.load_weights(f"./tf_models/{dirname}/model_weights.tf")
# except:
#   model.load_weights(f"../tf_models/{dirname}/model_weights.tf")

print(model.classify(testlst))
print(model._while_retrieve(testlst))

savedir = "test_saving_models/model"
model.save_weights(savedir)

del model
model = PaReNT.PaReNT(char_vocab=for_vocab,
                      train_len=len(train_df),
                      embedding_model=PaReNT.multibpe_model,
                      **parse_out_important(dirname))
model.load_weights(savedir, skip_mismatch=False)

print(model.classify(testlst))
print(model._while_retrieve(testlst))

model.save_weights(savedir)
del model
model = PaReNT.PaReNT(char_vocab=for_vocab,
                      train_len=len(train_df),
                      embedding_model=PaReNT.multibpe_model,
                      **parse_out_important(dirname))
model.load_weights(savedir)

print(model.classify(testlst))
print(model._while_retrieve(testlst))

model.save(savedir+"h5")