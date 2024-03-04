import argparse
import datetime
import os
import re
from tqdm import tqdm
import time
import sys
import time
import signal
import timeout_decorator

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from transformers import AutoTokenizer


parser = argparse.ArgumentParser(description='Model hyperparameters.')
parser.add_argument('--batch_size', type=int, default=128, help='Size of each batch in the epoch.')
parser.add_argument('--dim', type=int, default=64, help='Dimensionality of the RNN cell in both the decoder and encoder.')
parser.add_argument('--embed_dim', type=int, default=16, help='Dimensionality of the character embeddings.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--num_layers', type=int, default=1, help='Number of extra layers.')

args = parser.parse_args()
epoch = 0
now = datetime.datetime.now()
date = "-".join([str(now.year), str(now.month), str(now.day)])

print(f"{date}/e-{epoch+1}dim-{args.dim}numlayers-{args.num_layers}b-{args.batch_size}edim-{args.embed_dim}lr-{args.lr}")

# --- Define helper functions ---
def get_vocab(tensor: tf.RaggedTensor) -> list:
    tensor = tf.reshape(tensor, [-1])
    uniq = tf.unique(tensor)
    vocab = list(uniq[0].numpy())
    vocab.sort()

    vocab = [b"<pad>",  b"<start>", b"<end>"] + vocab
    return vocab

def lst_to_padded_tensor(_x: list, start_end_tokens: bool=True) -> (tf.Tensor, list):
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
    #FIXME:
    _df = _df.dropna()

    _x = list(_df['lexeme'])
    _x, _x_vocab = lst_to_padded_tensor(_x)

    _Y = list(_df['parents'])
    train_Y, _Y_vocab = lst_to_padded_tensor(_Y)

    _x_dataset = tf.data.Dataset.from_tensor_slices(_x)
    _Y_dataset = tf.data.Dataset.from_tensor_slices(train_Y)
    _dataset = tf.data.Dataset.zip((_x_dataset, _Y_dataset))

    ##TODO: The drop_remainder thing might cause problems, though it probably won't
    if subset == "train":
        _dataset = _dataset.batch(batch_size, drop_remainder=True)

    return _dataset, _x_vocab, _Y_vocab

def _parse_(output:list) -> str:
    special_tokens = [b"<pad>", b"<start>", b"<end>"]
    newlst = []
    for char in output:
        presence = bool(char not in special_tokens)
        if presence:
            newlst.append(char.numpy())
    return b''.join(newlst).decode("UTF-8")

@timeout_decorator.timeout(20)
def parse_output(tup, sanity_check=False):
    wordlist, results = tup

    results = [_parse_(i) for i in results]
    if sanity_check:
        for word, retrieval in zip(wordlist, results):
            print('Input: %s' % (word))
            print('Predicted retrieval: {}'.format(retrieval))
    else:
        return results
# ---



## --- Load data ---
BATCH_SIZE = args.batch_size
print(tf.config.experimental.list_physical_devices())

#Train
train_dataset, x_vocab, Y_vocab = load_data("train",
                                            batch_size=BATCH_SIZE)
num_examples = train_dataset.cardinality()

#Longest word lengths
max_length_input = train_dataset.element_spec[0].shape[-1]
max_length_output = train_dataset.element_spec[1].shape[-1]

#No. of unique chars
vocab_inp_size = len(x_vocab)
vocab_tar_size = len(Y_vocab)

#Dev
dev_dataset, dev_x_vocab, dev_Y_vocab = load_data("test",
                                                  batch_size=BATCH_SIZE)

#Example batches for Encoder and Decoder unit tests
example_input_batch, example_output_batch = [*train_dataset.take(1)][0]
# ---


# --- Model ---
print("Starting building the model...")
#TODO: Wrap into single PaReNT class
#TODO: Include training as a method of said class

#TODO: Add word embeddings, somehow

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, string_lookup_x):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.string_lookup_x = string_lookup_x

        ##-------- LSTM layer in Encoder ------- ##
        lstm_layer = tf.keras.layers.GRU(self.enc_units,
                                  return_sequences=True,
                                  return_state=True,
                                  recurrent_initializer='glorot_uniform')
        self.lstm_layer = tf.keras.layers.Bidirectional(lstm_layer,
                                                        merge_mode='sum')

        numlayers = args.num_layers
        layers = []
        gru = tf.keras.layers.GRU(self.enc_units,
                                  return_sequences=True,
                                  return_state=True,
                                  recurrent_initializer='glorot_uniform')
        bidir = tf.keras.layers.Bidirectional(gru,
                                              merge_mode='sum')

        for i in range(numlayers-1):
            layers.append(bidir)

        self.additional_layers = layers


    @tf.function
    def call(self, x, hidden):
        x = self.string_lookup_x(x)
        x = self.embedding(x)

        output, h, c = self.lstm_layer(x, initial_state=hidden)
        for layer in self.additional_layers:
            output, h, c = layer(output, initial_state=[h, c])
        return output, h, c

    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, string_lookup_Y, attention_type='luong'):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.attention_type = attention_type

        # # String lookup
        self.string_lookup_Y = string_lookup_Y

        # Embedding Layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # Final Dense layer on which softmax will be applied
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Define the fundamental cell for decoder recurrent structure
        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)

        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(self.dec_units,
                                                                  None, self.batch_sz * [max_length_input],
                                                                  self.attention_type)
        # Wrap attention mechanism with the fundamental rnn cell of decoder
        self.rnn_cell = self.build_rnn_cell(self.batch_sz)

        # Define the decoder with respect to fundamental rnn cell
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)

    def build_rnn_cell(self, batch_sz):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell,
                                                self.attention_mechanism, attention_layer_size=self.dec_units)
        return rnn_cell

    def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='luong'):
        # ------------- #
        # typ: Which sort of attention (Bahdanau, Luong)
        # dec_units: final dimension of attention outputs
        # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
        # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_input (for masking purpose)

        if (attention_type == 'bahdanau'):
            return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory,
                                                 memory_sequence_length=memory_sequence_length)
        else:
            return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory,
                                              memory_sequence_length=memory_sequence_length)

    def build_initial_state(self, batch_sz, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state

    def call(self, inputs, initial_state):
        x = self.string_lookup_Y(inputs)
        x = self.embedding(x)
        outputs, _, _ = self.decoder(x, initial_state=initial_state,
                                     sequence_length=self.batch_sz * [max_length_output - 1])
        return outputs

class PaReNT(tf.keras.Model):
    def __init__(self, BATCH_SIZE=BATCH_SIZE) -> None:
        super().__init__()

        # --- Model hyperparameters ---
        schedule = tf.keras.optimizers.schedules.CosineDecay(args.lr, decay_steps=1000,
                                                              alpha=0.0, name=None)

        self.optimizer = tf.keras.optimizers.Adam(args.lr)
        self.embedding_dim = args.embed_dim
        self.units = args.dim

        self.num_examples = num_examples
        # ---

        # --- Model layers ---
        self.string_lookup_x = tf.keras.layers.StringLookup(vocabulary=x_vocab)
        self.string_lookup_Y = tf.keras.layers.StringLookup(vocabulary=Y_vocab)
        self.inverse_string_lookup_Y = tf.keras.layers.StringLookup(vocabulary=Y_vocab, invert=True)

        self.encoder = Encoder(vocab_inp_size, self.embedding_dim,
                               self.units, BATCH_SIZE,
                               string_lookup_x=self.string_lookup_x)

        self.decoder = Decoder(vocab_tar_size, self.embedding_dim,
                               self.units, BATCH_SIZE,
                               string_lookup_Y=self.string_lookup_Y, attention_type='luong')



        # ---


    def loss_function(self, real, pred):
        # real shape = (BATCH_SIZE, max_length_output)
        # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        loss = cross_entropy(y_true=real, y_pred=pred)
        mask = tf.logical_not(tf.math.equal(real, 1))   #output 0 for y=0 else output 1
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = mask*loss
        loss = tf.reduce_mean(loss)
        return loss

    @tf.function
    def call(self, inp, training=False):
        enc_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_h, enc_c = self.encoder(inp, enc_hidden)

        if training == True:
            return enc_output, enc_h, enc_c
        else:
            return self.retrieve(inp)

    def train_step(self, dataset):
        inp, targ = dataset

        with tf.GradientTape() as tape:
            #TODO: Include decoder in the model call
            enc_output, enc_h, enc_c = self(inp, training=True)

            dec_input = targ[:, :-1]  # Ignore <end> token

            # Set the AttentionMechanism object with encoder_outputs
            self.decoder.attention_mechanism.setup_memory(enc_output)

            # Create AttentionWrapperState as initial_state for decoder
            decoder_initial_state = self.decoder.build_initial_state(BATCH_SIZE, [enc_h, enc_c], tf.float32)
            pred = self.decoder(dec_input, decoder_initial_state)
            logits = pred.rnn_output

            real = targ[:, 1:]  # ignore <start> token

            loss = self.loss_function(self.string_lookup_x(real), logits)

            variables = self.encoder.trainable_variables + self.decoder.trainable_variables
            gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))
        return {"loss": loss}

    def __evaluate_wordlist__(self, wordlist):
        if type(wordlist) == list:
            # TODO: Get rid of
            inputs, _ = lst_to_padded_tensor(wordlist)


            # # --- Padding. --- TODO: Try without, might be unnecessary
            # # TODO:(Also if necessary, it might also be necessary to truncate)
            # diff = (max_length_input - wordlist.shape[1])
            # batchsize = args.batch_size
            #
            # padding = tf.fill([batchsize, diff], "<pad>")
            # inputs = tf.concat([wordlist, padding], axis=1)
            # ---
        else:
            inputs = wordlist

        inference_batch_size = tf.shape(inputs)[0]

        enc_start_state = [tf.zeros((inference_batch_size, self.units)), tf.zeros((inference_batch_size, self.units))]
        enc_out, enc_h, enc_c = self.encoder(inputs, enc_start_state)

        dec_h = enc_h
        dec_c = enc_c

        start_tokens = tf.fill([inference_batch_size], string_lookup('<start>'))
        start_tokens = tf.cast(start_tokens, tf.int32)
        end_token = tf.cast(string_lookup('<end>'), tf.int32)

        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

        # Instantiate BasicDecoder object
        decoder_instance = tfa.seq2seq.BasicDecoder(cell=self.decoder.rnn_cell, sampler=greedy_sampler,
                                                    output_layer=self.decoder.fc)
        # Setup Memory in decoder stack
        self.decoder.attention_mechanism.setup_memory(enc_out)

        # set decoder_initial_state
        decoder_initial_state = self.decoder.build_initial_state(inference_batch_size, [enc_h, enc_c], tf.float32)

        ### Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder
        ### decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this.
        ### You only need to get the weights of embedding layer, which can be done by decoder.embedding.variables[0] and pass this callabble to BasicDecoder's call() function

        decoder_embedding_matrix = self.decoder.embedding.variables[0]

        outputs, _, _ = decoder_instance(decoder_embedding_matrix,
                                         start_tokens=start_tokens,
                                         end_token=end_token,
                                         initial_state=decoder_initial_state)
        return outputs.sample_id

    @timeout_decorator.timeout(20)
    def retrieve(self, wordlist):
        results = self.__evaluate_wordlist__(wordlist)
        results = self.inverse_string_lookup_Y(results)

        return wordlist, results



    # ---


# --- Checkpoint ---

#
# checkpoint_dir = 'models/' + date
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(optimizer=optimizer,
#                                  encoder=encoder,
#                                  decoder=decoder)
# ---


#TODO: Include as part of the model
string_lookup = tf.keras.layers.StringLookup(vocabulary=Y_vocab)
inverse_string_lookup = tf.keras.layers.StringLookup(vocabulary=Y_vocab, invert=True)



# --- Training ---
model_examples = ["bába", "enchantenment", "deathhead", "Bonbondieb", "černobřichý", "черний"]
print(model_examples)
#retrieve(model_examples)

EPOCHS = 20
model = PaReNT()
model.compile(optimizer=model.optimizer,
              loss=model.loss_function,
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

for epoch in tqdm(range(EPOCHS)):
    start = time.time()
    model.fit(train_dataset,
              epochs=1)

    try:
        print(parse_output(model.retrieve(model_examples), sanity_check=True))
    except:
        print("Shit got stuck")
    finally:
        if (epoch+1) % 2 == 0:
            now = datetime.datetime.now()
            date = "-".join([str(now.year), str(now.month), str(now.day)])

            file_path = f"../models/{date}/e-{epoch+1}dim-{args.dim}numlayers-{args.num_layers}b-{args.batch_size}edim-{args.embed_dim}lr-{args.lr}/model"

            print("Saving model...")
            model.save_weights(file_path)
            print("Model weights saved!")

# ---

print("Done!")

# --- Inference (sanity check) ---
#TODO: Implement beam search decodinsdddd

# ---

