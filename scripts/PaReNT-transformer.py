import argparse
import datetime
import time
import os
from tqdm import tqdm
import shutil

import timeout_decorator

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import pandas as pd
import tensorflow as tf
import trax
import jax
from trax.supervised import training
from trax import layers as tl

from trax import fastmath
from trax.fastmath import numpy as np

from eval_PaReNT_transformer import eval_model

print("Confirm that TF can't and JAX can see the GPU")
print("TF:")
print(tf.config.get_visible_devices())
print("JAX:")
print(jax.devices())

trax.fastmath.ops.use_backend("tf")
print("backend set")
print(trax.fastmath.ops.backend_name())

parser = argparse.ArgumentParser(description='Model hyperparameters.')
parser.add_argument('--batch_size', type=int, default=256, help='Size of each batch in the epoch.')
parser.add_argument('--epochs', type=int, default=20, help='Size of each batch in the epoch.')
parser.add_argument('--dim', type=int, default=256, help='Dimensionality of FC layers in the Transformer.')
parser.add_argument('--embed_dim', type=int, default=128,
                    help='Dimensionality of everything in the LSTM and most everything in the Transformer.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropoutembed.')
parser.add_argument('--n_encoder_layers', type=int, default=2, help='Number of encoder layers.')
parser.add_argument('--n_decoder_layers', type=int, default=1, help='Number of decoder layers.')
parser.add_argument('--n_heads', type=int, default=1, help='Number of attention heads. Should be a power of two, ideally.')
parser.add_argument('--model_type', type=str, default="lstm", help="Type of model")
parser.add_argument('--test_mode', type=int, default=0, help="Whether to run with low number of batches")

args = parser.parse_args()

epoch = 0
now = datetime.datetime.now()
date = "-".join([str(now.year), str(now.month), str(now.day)])

modelname = f"{args.model_type}-{date}e-{args.epochs}dim-{args.dim}" \
            f"n_e_layers-{args.n_encoder_layers}n_d_layers-{args.n_encoder_layers}" \
            f"b-{args.batch_size}edim-{args.embed_dim}lr-{args.lr}"

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    print("Memory limit set")
  except RuntimeError as e:
    print(e)

# --- Define helper functions ---
#Lookup layers
def get_vocab(tensor: tf.RaggedTensor) -> list:
    tensor = tf.reshape(tensor, [-1])
    uniq = tf.unique(tensor)
    vocab = list(uniq[0].numpy())
    vocab.sort()

    vocab = [b"<pad>"] + vocab
    return vocab

def lst_to_padded_tensor(_x: list, start_end_tokens: bool=False) -> (tf.Tensor, list):
    _x = tf.strings.unicode_split([i for i in _x], 'UTF-8')
    _x_vocab = get_vocab(_x)

    if start_end_tokens:
        _start_tokens = tf.fill([_x.bounding_shape()[0], 1], b"<start>")
        _end_tokens = tf.fill([_x.bounding_shape()[0], 1], b"<end>")

        _x = tf.concat([_start_tokens, _x], axis=1)
        _x = tf.concat([_x, _end_tokens], axis=1)

    _x = _x.to_tensor(default_value=b"<pad>")

    return _x, _x_vocab

def load_data(subset: str, batch_size: int=64) -> (tf.data.Dataset, list, list):
    assert subset in ["test", "train", "validate"]
     #global _df

    print(f"Loading {subset} data...")
    pdlist = []
    file_path = f"data/{subset}/"

    if not os.path.exists(file_path):
        file_path = "../" + file_path

    for i in tqdm(os.listdir(file_path)):
        df = pd.read_csv(file_path+i, sep=".")
        pdlist.append(df)
    _df = pd.concat(pdlist)
    del pdlist

    _df = _df.sample(frac=1)
    print(f"Length of undropped df: {len(_df)}")
    #FIXME:
    _df = _df.dropna()
    print(f"Length of dropped df: {len(_df)}")

    _x = list(_df['lexeme'])
    _x, _x_vocab = lst_to_padded_tensor(_x)

    _Y = list(_df['parents'])
    train_Y, _Y_vocab = lst_to_padded_tensor(_Y)

    ##Weights
    _W = tf.ones([len(_Y), 1])

    _x_dataset = tf.data.Dataset.from_tensor_slices(_x)
    _Y_dataset = tf.data.Dataset.from_tensor_slices(train_Y)
    _W_dataset = tf.data.Dataset.from_tensor_slices(_W)
    _dataset = tf.data.Dataset.zip((_x_dataset, _Y_dataset, _W_dataset))

    #TODO: The drop_remainder thing might cause problems, though it probably won't
    if batch_size is not None:
        _dataset = _dataset.batch(batch_size, drop_remainder=True)

    return _dataset, _x_vocab, _Y_vocab

def _parse_(output:list) -> str:
    special_tokens = [b"<pad>"]
    newlst = []
    for char in output:
        presence = bool(char.numpy() not in special_tokens)
        if presence:
            newlst.append(char.numpy())
    return b''.join(newlst).decode("UTF-8")

def parse_output(tup, sanity_check=False):
    wordlist, results = tup

    results = [_parse_(i) for i in results]
    if sanity_check:
        for word, retrieval in zip(wordlist, results):
            print('Input: %s' % (word))
            print('Predicted retrieval: {}'.format(retrieval))
    else:
        return results

def encode_into_integerIDs(x, vocab):
    lookup_layer = tf.keras.layers.StringLookup(vocabulary=vocab)

    return lookup_layer(x) - 1

def decode_from_integerIDs(y, vocab):
    lookup_layer = tf.keras.layers.StringLookup(vocabulary=vocab, invert=True)

    return lookup_layer(y+1)

def masked_cat_crossentropy_function(pred, real):
    # real shape = (BATCH_SIZE, max_length_output)
    # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
    cross_entropy = trax.layers.metrics.WeightedCategoryCrossEntropy()
    loss = cross_entropy(pred, real, 1)
    mask = np.logical_not(real == 0)   #output 0 for y=0 else output 1
    loss = mask*loss
    loss = np.mean(loss)
    return loss

#loss_function = trax.layers.base.Fn("loss", masked_cat_crossentropy_function, n_out=1)
# ---



## --- Load data ---
BATCH_SIZE = args.batch_size
print(tf.config.experimental.list_physical_devices())

#Train
train_dataset, x_vocab, Y_vocab = load_data("train",
                                            batch_size=BATCH_SIZE)
num_batches = train_dataset.cardinality()

#Longest word lengths
max_length_input = train_dataset.element_spec[0].shape[-1]
max_length_output = train_dataset.element_spec[1].shape[-1]

#No. of unique chars
vocab_inp_size = len(x_vocab)
vocab_tar_size = len(Y_vocab)

#Dev
dev_dataset, _, _ = load_data("test",
                              batch_size=BATCH_SIZE)

#Example batches for Encoder and Decoder unit tests
example_input_batch, example_output_batch, weight = [*train_dataset.take(1)][0]

train_dataset = train_dataset.map(lambda a, b, c: (encode_into_integerIDs(a, x_vocab),
                                                   encode_into_integerIDs(b, Y_vocab),
                                                   c))
dev_dataset = dev_dataset.map(lambda a, b, c: (encode_into_integerIDs(a, x_vocab),
                                               encode_into_integerIDs(b, Y_vocab),
                                               c))
# ---

# --- Model building functions ---
def build_transformer(x_vocab: list, Y_vocab: list, d_model: int, n_encoder_layers,
                      n_decoder_layers, n_heads, d_ff, dropout, mode):
    model = trax.models.transformer.Transformer(input_vocab_size=len(x_vocab),
                                                output_vocab_size=len(Y_vocab),
                                                d_model=d_model,
                                                d_ff=d_ff,
                                                n_encoder_layers=n_encoder_layers,
                                                n_decoder_layers=n_decoder_layers,
                                                n_heads=n_heads,
                                                dropout=dropout,
                                                dropout_shared_axes=(0,1),
                                                mode=mode,
                                                )
    return model

def build_lstm(x_vocab: list, Y_vocab: list, d_model, n_encoder_layers,
               n_decoder_layers, n_heads, dropout, mode):
    model = trax.models.rnn.LSTMSeq2SeqAttn(input_vocab_size=len(x_vocab),
                                            target_vocab_size=len(Y_vocab),
                                            d_model=d_model,
                                            n_encoder_layers=n_encoder_layers,
                                            n_decoder_layers=n_decoder_layers,
                                            n_attention_heads=n_heads,
                                            attention_dropout=dropout,
                                            mode=mode)
    return model

def build_model(model_type, mode):
    if model_type.casefold() == "transformer":
        model = build_transformer(x_vocab,
                                  Y_vocab,
                                  d_model=args.embed_dim,
                                  n_encoder_layers=args.n_encoder_layers,
                                  n_decoder_layers=args.n_decoder_layers,
                                  n_heads=args.n_heads,
                                  d_ff=args.dim,
                                  dropout=args.dropout,
                                  mode=mode)

        return model
    elif model_type.casefold() == "lstm":
        model = build_lstm(x_vocab,
                           Y_vocab,
                           d_model=args.embed_dim,
                           n_encoder_layers=args.n_encoder_layers,
                           n_decoder_layers=args.n_decoder_layers,
                           n_heads=args.n_heads,
                           dropout=args.dropout,
                           mode=mode)

        return model
    else:
        raise ValueError("Model type not understood")

# ---

# --- Building model ---
model = build_model(args.model_type, mode="train")
# ---

#train_dataset_trax = np.nditer([*train_dataset.as_numpy_iterator()]
#dev_dataset_trax = np.nditer([*dev_dataset.as_numpy_iterator()])


# # --- Specifying training rubbish ---
# # Training task.
# train_task = training.TrainTask(
#     labeled_data=train_dataset.as_numpy_iterator(),
#     loss_layer=tl.metrics.WeightedCategoryCrossEntropy(),
#     optimizer=trax.optimizers.Adam(args.lr),
# )
#
#
# # Evaluation task.
# eval_task = training.EvalTask(
#     labeled_data=dev_dataset.as_numpy_iterator(),
#     metrics=[tl.metrics.WeightedCategoryCrossEntropy(), tl.WeightedCategoryAccuracy(), tl.MaskedSequenceAccuracy()],
#     n_eval_batches=20  # For less variance in eval numbers.
# )
#
# # Training loop saves checkpoints to output_dir.
#
# print("Training loop ready")

EPOCHS = args.epochs

model_examples = ["bába", "enchantment", "deathhead", "Bonbondieb", "černobřichý", "черний"]
print(f'Sanity check examples: {model_examples}')

# inp, _ = lst_to_padded_tensor(model_examples)
# sanity_batch_size = int(inp.shape[0])
# print(inp)
# print(sanity_batch_size)
# inp = encode_into_integerIDs(inp, x_vocab)
# print(inp.numpy())
# retrieval = trax.supervised.decoding.autoregressive_sample(model,
#                                                            inp.numpy(),
#                                                            temperature=0.0,
#                                                            batch_size=sanity_batch_size,
#                                                            max_length=100)

# inverted_lookup = tf.keras.layers.StringLookup(vocabulary=Y_vocab,
#                                                invert=True)
# print(parse_output((model_examples, inverted_lookup(retrieval))))
# ---

# --- Run Training ---
print(f"Epochs: {EPOCHS}")
print(f"Batches: {num_batches}")
filepath = '../models/'
start = time.time()

for epoch in tqdm(range(EPOCHS)):
    epoch_filepath = filepath + modelname + f"epoch-{epoch}" + str(start)
    if os.path.exists(epoch_filepath):
        shutil.rmtree(epoch_filepath)

    checkpoint_at = 699 if args.test_mode == 1 else int(num_batches - ((int(num_batches) % 100) + 1))-1

    # Training task.
    train_task = training.TrainTask(
        labeled_data=train_dataset.as_numpy_iterator(),
        loss_layer=tl.metrics.WeightedCategoryCrossEntropy(),
        optimizer=trax.optimizers.Adam(args.lr),
        n_steps_per_permanent_checkpoint=checkpoint_at
    )

    # Evaluation task.
    eval_task = training.EvalTask(
        labeled_data=dev_dataset.as_numpy_iterator(),
        metrics=[tl.metrics.WeightedCategoryCrossEntropy(), tl.WeightedCategoryAccuracy(), tl.MaskedSequenceAccuracy()],
        n_eval_batches=20,  # For less variance in eval numbers.
        )

    os.mkdir(epoch_filepath)

    training_loop = training.Loop(model,
                                  train_task,
                                  eval_tasks=eval_task,
                                  output_dir=epoch_filepath)

    if args.test_mode == 1:
        print("!TESTMODE!")

        training_loop.run(700)
    else:
        training_loop.run(num_batches - ((int(num_batches) % 100) + 1))



    #model.save_to_file(epoch_filepath, weights_only=True)


    # inp, _ = lst_to_padded_tensor(model_examples, start_end_tokens=False)
    # sanity_batch_size = int(inp.shape[0])
    # inp = encode_into_integerIDs(inp, x_vocab).numpy()
    # # width = inp.shape[-1]
    # # max_width = train_dataset.element_spec[0].shape[-1]
    # # padding = np.zeros([sanity_batch_size, max_width-width], dtype=int)
    # # inp = np.concatenate([inp.numpy(), padding],axis=1)
    # print(inp)
    # print(inp.shape)
    # retrieval = trax.supervised.decoding.autoregressive_sample(model_predict,
    #                                                            inp,
    #                                                            temperature=0.0,
    #                                                            batch_size=sanity_batch_size,
    #                                                            max_length=100)
    # print(parse_output((model_examples, decode_from_integerIDs(retrieval, Y_vocab))))




    model_predict = build_model(args.model_type, mode="predict")
    model_predict.init_from_file(epoch_filepath + "/model.pkl.gz", weights_only=True)

    frac = 0.01
    print(f"Evaluating...Fraction{frac}")
    eval_model(model_predict, "../data/test/", epoch_filepath, x_vocab, Y_vocab, frac=frac)
        # ---