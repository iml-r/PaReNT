import os
import sys

import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

tfa.options.disable_custom_kernel()

try:
    from scripts.PaReNT_utils import grapheme_encode, cyril_to_latin, latin_to_cyril, decide_word_origin, retrieval_accuracy, contains_empty_lst, parse_out_important
    #import scripts.derinet
except:
    try:
        from .PaReNT_utils import grapheme_encode, cyril_to_latin, latin_to_cyril, decide_word_origin, retrieval_accuracy, contains_empty_lst, parse_out_important
    except:
        from PaReNT_utils import grapheme_encode, cyril_to_latin, latin_to_cyril, decide_word_origin, retrieval_accuracy, contains_empty_lst, parse_out_important
    #import derinet

from tqdm import tqdm
from bpemb import BPEmb
import time
from typing import List, Dict, Tuple, Any, Literal
import argparse
from collections import Counter
from itertools import product
import numpy as np

from sklearn.metrics import balanced_accuracy_score, accuracy_score
from datetime import date
import prettytable as pt
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.cluster.hierarchy as h



Vector = List[float]
StringVec = List[str]
IntVec = List[int]
TupleList = List[Tuple[str, str]]

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
print(tf.config.list_physical_devices('GPU'))

sns.set_theme(palette="colorblind")

max_length_input = 65
max_length_output = 100


def bpe_encode(x: str, model: BPEmb) -> np.ndarray:
    """
    Encodes the string using the given bpeMB model.
    Returns a numpy array of non-recursively BPE-encoded string pieces,
    without the special token marking the beginning of a word,
    since the task runs on isolated lexemes.
    """

    piece_list = model.spm.encode_as_pieces(x)
    # Get rid of the annoying symbol
    piece_list[0] = piece_list[0][1:]

    while '' in piece_list:
        piece_list.remove('')

    return np.array(piece_list, dtype=str)


def PaReNT_encode_tf(lang_code: str, lexeme: str, model: BPEmb) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Accepts:

    1. a special language token representing one of
    the seven languages in scope,
    2. a lexeme in its dictionary-standard form, and
    3. a bpeMB multilingual model.

    Returns:

    1. The given language's token;
    2. The lexeme represented as a Ragged Tensor
    representing each subword of the given lexeme split
    into characters, shape [length_of_lexeme, None]; and
    3. The lexeme represented as a 300-dimensional embedding
    of the same subword sequence, shaped [length_of_lexeme, 300]
    """

    subwords = bpe_encode(lexeme, model)

    if lang_code == "ru":
        characterwise = [np.array(cyril_to_latin(subword).split(" ")) for subword in grapheme_encode(subwords.tolist())]
    else:
        characterwise = [np.array(subword.split(" ")) for subword in grapheme_encode(subwords.tolist())]

    semantic = []
    for subword in subwords:
        id = model.spm.PieceToId(subword)
        semantic.append(model.emb.vectors[id])

    return lang_code, np.array(characterwise, dtype=object), np.array(semantic)


def load_data(df: pd.DataFrame,
              model: BPEmb,
              start: float,
              end: float):
    len_df = len(df)
    start = int(start * len_df)
    end = int(end * len_df)

    df = df.iloc[start:end]

    lang_list = []
    characterwise_list = []
    semantic_list = []

    parent_label_list = []
    class_label_list = []

    classifier_dict = {"Unmotivated": 0, "Derivative": 1, "Compound": 2}

    for row in df.iterrows():
        row = row[1]
        lexeme = row["lexeme"]

        lang, characterwise, semantic = PaReNT_encode_tf(lang_code=row["language"],
                                                         lexeme=lexeme,
                                                         model=model)

        lang_list.append(lang)
        characterwise_list.append(characterwise)
        semantic_list.append(semantic)

        lexeme_class = classifier_dict[decide_word_origin([lexeme], [row["parents"]])[0]]
        class_label_list.append(lexeme_class)

        parents = row["parents"]
        if lang == "ru":
            parents = cyril_to_latin(parents)
        parents = tf.constant(["[START]"] + grapheme_encode(parents).split(" ") + ["[END]"])
        parent_label_list.append(parents)

    x1 = tf.data.Dataset.from_tensor_slices(lang_list).map(lambda language_label: tf.reshape(language_label, [1]))
    x2 = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(characterwise_list))
    x3 = tf.data.Dataset.from_tensor_slices(tf.keras.preprocessing.sequence.pad_sequences(semantic_list,
                                                                                          padding='post',
                                                                                          maxlen=65,
                                                                                          value=0.,
                                                                                          dtype=float
                                                                                          ))

    y_class = tf.data.Dataset.from_tensor_slices(class_label_list)
    y_retrieve = tf.data.Dataset.from_tensor_slices(tf.keras.preprocessing.sequence.pad_sequences(parent_label_list,
                                                                                                  padding='post',
                                                                                                  maxlen=100 + 1,
                                                                                                  value="<pad>",
                                                                                                  dtype=object))

    return tf.data.Dataset.zip(((x1, x2, x3), (y_retrieve, y_class))), characterwise_list


def load_data_chunk(model: BPEmb,
                    subset: str = "train",
                    cluster: bool = False,
                    test_mode: bool = False,
                    threshold: float = 0.01) -> Tuple[tf.data.Dataset, StringVec]:
    if cluster:
        prefix = "../"
    else:
        prefix = ""
    directory = f"{prefix}data/{subset}/"

    dflist = []
    for datasource_name in tqdm(os.listdir(directory), desc=f"Loading {subset} CSV files", colour='#e6af00'):
        df = pd.read_csv(directory + datasource_name, sep=".")
        dflist.append(df)

    df = pd.concat(dflist)
    df = df.dropna()

    len_df = len(df)

    if test_mode:
        df = df.sample(frac=threshold)

    dataset, for_vocab = load_data(df=df,
                                   model=model,
                                   start=0.,
                                   end=threshold)
    for_vocab = []
    arange = np.arange(threshold, 1, threshold)
    arange = tqdm(arange, desc=f"Building {subset} TensorFlow dataset in {len(arange) + 1} chunks", colour="#a8e4ea",
                  leave=None)

    for start in arange:
        dataset2, vocab = load_data(df=df,
                                    model=model,
                                    start=start,
                                    end=start + threshold)
        dataset = dataset.concatenate(dataset2)
        for_vocab += vocab

    for_vocab = [i for y in for_vocab for i in y.flatten()]
    for_vocab = [i for y in for_vocab for i in y] + ["_", "[START]", "[END]"]

    print(f"Data from {subset} loaded!")
    return dataset, for_vocab


def preprocess_data(df: pd.DataFrame,
                    model: BPEmb,
                    start: float,
                    end: float):
    len_df = len(df)
    start = int(start * len_df)
    end = int(end * len_df)

    df = df.iloc[start:end]

    lang_list = []
    characterwise_list = []
    semantic_list = []

    class_label_list = []
    parent_label_list = []

    classifier_dict = {"Unmotivated": 0, "Derivative": 1, "Compound": 2}

    for row in df.iterrows():
        row = row[1]
        lexeme = row["lexeme"]

        lang, characterwise, semantic = PaReNT_encode_tf(lang_code=row["language"],
                                                         lexeme=lexeme,
                                                         model=model)

        lang_list.append(lang)
        characterwise_list.append(characterwise)
        semantic_list.append(semantic)

        lexeme_class = classifier_dict[decide_word_origin([lexeme], [row["parents"]])[0]]
        class_label_list.append(lexeme_class)

        parents = row["parents"]
        if lang == "ru":
            parents = cyril_to_latin(parents)
        parents = ["[START]"] + grapheme_encode(parents).split(" ") + ["[END]"]
        parent_label_list.append(parents)

    x1 = tf.reshape(tf.constant(lang_list), [-1, 1])
    x2 = tf.ragged.constant(characterwise_list)
    x3 = tf.keras.preprocessing.sequence.pad_sequences(semantic_list,
                                                       padding='post',
                                                       maxlen=65,
                                                       value=0.,
                                                       dtype=float
                                                       )

    y_class = tf.constant(class_label_list)
    y_retrieve = tf.ragged.constant(parent_label_list)

    return ((x1, x2, x3), (y_retrieve, y_class)), characterwise_list

def data_generator(df: pd.DataFrame,
                   model: BPEmb,
                   subset: str = "train",
                   batch_size: int = 32):
    threshold = batch_size / len(df)

    arange = np.arange(0, 1, threshold)

    for start in arange:
        chunk, _ = preprocess_data(df=df,
                                   model=model,
                                   start=start,
                                   end=start + threshold)
        yield chunk


def load_df(subset: str = "train",
            cluster: bool = True):
    if cluster:
        prefix = "../"
    else:
        prefix = ""
    directory = f"{prefix}data/{subset}/"

    dflist = []
    for datasource_name in tqdm(os.listdir(directory), desc=f"Loading {subset} CSV files", colour='#e6af00'):
        df = pd.read_csv(directory + datasource_name, sep=".")
        col = np.repeat(datasource_name, len(df))
        df["dataset_name"] = col
        dflist.append(df)

    df = pd.concat(dflist)
    df = df.dropna()

    if subset == "train":
        df = df.sample(frac=1)

    return df




def get_vocab(df: pd.DataFrame,
              subset: str = "train"):
    for_vocab = list(df["lexeme"]) + list(df["parents"])

    for_vocab = sorted(Counter([cyril_to_latin(i) for y in for_vocab for i in y]).items(),
                       key=lambda a: (a[0], a[1]))
    for_vocab = [i[0] for i in for_vocab if i[1] > 20] + ["_", "[START]", "[END]"]

    print(f"Loading {subset} vocab done!")

    return for_vocab


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# train_data, for_vocab = load_data_chunk(multibpe_model,
#                                         test_mode=False)

def debug_print(x):
    print(x)
    tf.print(x)


print("Loading model for static embeddings...")
multibpe_model = BPEmb(lang='multi',
                       vs=100000,
                       dim=300)


preprocessed_input_signature = [tf.TensorSpec(shape=[None, 1], dtype=object), tf.RaggedTensorSpec(shape=[None, None, None], dtype=object), tf.RaggedTensorSpec(shape=[None, None, 300], dtype=tf.float32)]
hidden_signature = [[tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), tf.TensorSpec(shape=[None, None], dtype=tf.float32), tf.TensorSpec(shape=[None, None], dtype=tf.float32)], tf.TensorSpec([None, 2], dtype=tf.float32)]

#@tf.keras.saving.register_keras_serializable ## 1
class TransformerLikeBlock(tf.keras.Model):
    def __init__(self, units, attention_units, dropout,
                 activation=tf.nn.relu, **kwargs): ## 2
        super(TransformerLikeBlock, self).__init__(**kwargs) ## 3

        ## 4
        self.units = units
        self.attention_units = attention_units
        self.dropout = dropout
        self.activation = activation

        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=2,
                                                            key_dim=self.attention_units)
        self.add = tf.keras.layers.Add()
        self.layer_norm1 = tf.keras.layers.LayerNormalization()

        self.fc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.units,
                                                   activation=self.activation))
        self.dropout = tf.keras.layers.Dropout(self.dropout)
        self.layer_norm2 = tf.keras.layers.LayerNormalization()

    def get_config(self): ## 5
        config = super(TransformerLikeBlock, self).get_config()

        config["units"] = self.units
        config["attention_units"] = self.attention_units
        config["dropout"] = self.dropout
        config["activation"] = self.activation

        return config


    def call(self, inputs, training: bool = False, *args, **kwargs):
        attended = self.attention(inputs, inputs)
        added = self.add([inputs, attended])
        added = self.layer_norm1(added)

        fced = self.fc(added)
        fced = self.dropout(fced, training=training)

        added = self.add([fced, added])
        added = self.layer_norm2(added)
        return added

#@tf.keras.saving.register_keras_serializable ## 1
class LanguageEmbedding(tf.keras.layers.Layer):
    def __init__(self, **kwargs): ## 2
        super(LanguageEmbedding, self).__init__(**kwargs) ## 3

        langlist = ["en", "de", "nl", "es", "fr", "cs", "ru"] ## 4 -- hardcoded

        self.lang_code_lookup = tf.keras.layers.StringLookup(vocabulary=langlist,
                                                             name="lang_lookup")
        self.lang_embedding = tf.keras.layers.Embedding(input_dim=len(langlist) + 1,
                                                        output_dim=2,
                                                        name="lang_embed")
        self.lang_embedding_reshape = tf.keras.layers.Reshape((2,))

    def get_config(self): ## 5
        config = super(LanguageEmbedding, self).get_config()

        return config

    def call(self, inputs, *args, **kwargs):
        looked_up = self.lang_code_lookup(inputs)
        embedded = self.lang_embedding(looked_up)
        reshaped = self.lang_embedding_reshape(embedded)

        return reshaped

#@tf.keras.saving.register_keras_serializable ## 1
class Encoder(tf.keras.Model):
    def __init__(self, enc_units, attention_units, lookup_chars,
                 char_embedding, dropout, recurrent_dropout,
                 batch_sz, l=None, transblocks=0,
                 activation=tf.nn.relu, **kwargs): ## 2
        super(Encoder, self).__init__(**kwargs) ## 3

        self.batch_sz = batch_sz ## 4
        self.enc_units = enc_units
        self.l = l
        self.attention_units = attention_units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.transblocks = transblocks
        self.lookup_chars = lookup_chars
        self.char_embedding = char_embedding
        self.activation = activation


        bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=300,
                                                                    #recurrent_dropout=recurrent_dropout,
                                                                    kernel_regularizer=self.l),
                                               merge_mode='sum',
                                               name="encoder_char_lstm")
        self.char_lstm = tf.keras.layers.TimeDistributed(bilstm,
                                                         name="character_level_birnn")

        self.output_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=self.enc_units,
                                                                              return_sequences=True,
                                                                              return_state=True,
                                                                              #recurrent_dropout=self.recurrent_dropout,
                                                                              kernel_regularizer=self.l),
                                                         merge_mode='sum',
                                                         name="encoder_output_lstm")

        self.concat = tf.keras.layers.Concatenate(axis=1)
        #self.layer_norm = tf.keras.layers.LayerNormalization()

        self.transblock_stack = tf.keras.Sequential([TransformerLikeBlock(self.enc_units, self.attention_units, activation=activation, dropout=dropout) for i in range(transblocks)])

        self.lang_embed_spread = tf.keras.layers.Dense(units=300,
                                                       activation=None)

    def get_config(self): ## 5
        config = super(Encoder, self).get_config()

        config["batch_sz"] = self.batch_sz
        config["enc_units"] = self.enc_units
        config["l"] = self.l
        config["attention_units"] = self.attention_units
        config["dropout"] = self.dropout
        config["recurrent_dropout"] = self.recurrent_dropout
        config["transblocks"] = self.transblocks
        config["lookup_chars"] = self.lookup_chars
        config["char_embedding"] = self.char_embedding
        config["activation"] = self.activation

        return config

    def call(self, char_inputs, semant_inputs, hidden, lang_embeddings, use_lang_embeddings,
             neutralize_chars: bool, neutralize_semant: bool, training=False, *args, **kwargs):

        char_inputs = self.lookup_chars(char_inputs)
        char_inputs = self.char_embedding(char_inputs)
        char = self.char_lstm(char_inputs, training=training)

        char = char.to_tensor()
        char = char * tf.cast(not neutralize_chars, char.dtype)

        semant_inputs = semant_inputs * tf.cast(not neutralize_semant, semant_inputs.dtype)

        #Everything here has to be [batch_sz, None (concatentation dimension), 300]
        if use_lang_embeddings:
            lang_embeddings_spread = self.lang_embed_spread(lang_embeddings)
            lang_embeddings_expanded = tf.expand_dims(lang_embeddings_spread, axis=1)

            concatenated = self.concat([char, semant_inputs, lang_embeddings_expanded])

        else:
            concatenated = self.concat([char, semant_inputs])

        mask = tf.reduce_any(tf.not_equal(concatenated, 0.), axis=-1)

        outputs, inner_h, inner_c, h, c = self.output_lstm(concatenated,
                                                           initial_state=hidden,
                                                           mask=mask,
                                                           training=training)
        outputs = self.transblock_stack(outputs)

        return outputs, h, c

    def initialize_hidden_state(self):
        hidden = tf.zeros((self.batch_sz, self.enc_units))
        return [hidden, hidden, hidden, hidden]

#@tf.keras.saving.register_keras_serializable ## 1

class Classifier(tf.keras.Model):
    def __init__(self, fc_size, attention_units, dropout, activation=tf.nn.relu, **kwargs): ## 2
        super(Classifier, self).__init__(**kwargs) ## 3

        self.fc = fc_size ## 4
        self.attention_units = attention_units
        self.dropout = dropout
        self.activation = activation

        self.attention = tfa.layers.MultiHeadAttention(head_size=attention_units, num_heads=2)
        self.avgpool = tf.keras.layers.GlobalAvgPool1D()
        self.fc = tf.keras.layers.Dense(fc_size,
                                        activation=activation)
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.output_layer = tf.keras.layers.Dense(3,
                                                  activation=tf.nn.softmax)
        self.add_layer = tf.keras.layers.Add()

        self.transblock = TransformerLikeBlock(fc_size,
                                               attention_units,
                                               dropout=dropout,
                                               activation=activation)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def get_config(self): ## 5
        config = super(Classifier, self).get_config()

        config["fc_size"] = self.fc_size
        config["attention_units"] = self.attention_units
        config["dropout"] = self.dropout
        config["activation"] = self.activation

        return config

    @tf.function(reduce_retracing=True)
    def call(self, encoded, lang_embeddings, training: bool = False):
        attended = self.transblock(encoded, training=training)
        pooled = self.avgpool(attended)
        pooled = self.fc(pooled)
        pooled = self.dropout(pooled, training=training)
        concatenated = self.concat([pooled, lang_embeddings])
        return self.output_layer(concatenated)

#@tf.keras.saving.register_keras_serializable ## 1
class Decoder(tf.keras.Model):
    def __init__(self, char_embedding: tf.keras.layers.Layer, dec_units: int, batch_sz: int, vocab_size: int,
                 max_length_input: int = 65, max_length_output: int = 100, attention_type: str = 'luong',
                 activation=tf.nn.relu, **kwargs): ## 2
        super(Decoder, self).__init__(**kwargs) ## 3
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.batch_sz = batch_sz
        self.vocab_size = vocab_size
        self.max_length_input = max_length_input
        self.max_length_output = max_length_output
        self.attention_type = attention_type
        self.activation = activation

        # Embedding Layers
        self.char_embedding = char_embedding

        # Layer between Decoder RNN output (which needs to have the same dim as the final output) and lang embed concat
        self.fc = tf.keras.layers.Dense(vocab_size + 2,
                                        activation=None)

        # Final  layer on which softmax will be applied
        # self.final = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size + 2,
        #                                                                    activation=tf.nn.softmax))
        # self.concat = tf.keras.layers.Concatenate(axis=-1)

        # Define the fundamental cell for decoder recurrent structure
        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units,
                                                         activation=activation)

        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(self.dec_units,
                                                                  None,
                                                                  self.batch_sz * [max_length_input],
                                                                  self.attention_type)

        # Wrap attention mechanism with the fundamental rnn cell of decoder
        self.rnn_cell = self.build_rnn_cell(batch_sz=self.batch_sz)

        # Define the decoder with respect to fundamental rnn cell
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell,
                                                sampler=self.sampler,
                                                output_layer=self.fc)


    def get_config(self): ## 5
        config = super(Decoder, self).get_config()

        config["batch_size"] = self.batch_sz
        config["dec_units"] = self.dec_units
        config["batch_sz"] = self.batch_sz
        config["vocab_size"] = self.vocab_size
        config["max_length_input"] = self.max_length_input
        config["max_length_output"] = self.max_length_output
        config["attention_type"] = self.attention_type
        config["activation"] = self.activation

        return config

    def build_rnn_cell(self, batch_sz):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell,
                                                self.attention_mechanism,
                                                attention_layer_size=self.dec_units)
        return rnn_cell

    def build_attention_mechanism(self,
                                  dec_units,
                                  memory,
                                  memory_sequence_length,
                                  attention_type='luong'):
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

    def call(self, inputs, initial_state, lang_embeddings):
        x = self.char_embedding(inputs)
        seq_len = tf.cast(x.row_lengths(), tf.int32)
        x = x.to_tensor()


        outputs, _, _ = self.decoder(x, initial_state=initial_state,
                                     sequence_length=seq_len)

        outputs = outputs.rnn_output

        # lang_embeddings = tf.expand_dims(lang_embeddings, axis=1)
        # correct_shape = [tf.shape(outputs)[0], tf.shape(outputs)[1], 2]
        # lang_embeddings = tf.broadcast_to(lang_embeddings, correct_shape)
        #
        # concatenated = self.concat([outputs, lang_embeddings])
        #outputs = self.final(outputs)
        return outputs

#@tf.keras.saving.register_keras_serializable ## 1
class PaReNT(tf.keras.Model):
    def __init__(self, dirname: str = None,
                 char_vocab: List[str] = None,
                 batch_sz: int = 64,
                 embedding_model=multibpe_model,
                 train_len=1,
                 attention_units=256,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 char_embedding_dim=32,
                 units=512,
                 max_length_input=65,
                 max_length_output=100,
                 lr=1e-3,
                 transblocks=0,
                 activation=tf.nn.swish,
                 optimizer:str="Adam",
                 neutralize_chars:bool=False,
                 neutralize_semant:bool=False,
                 use_lang_embeddings: bool=True,
                 semantic_warmup: int=0): ## 2
        super(PaReNT, self).__init__() ## 3

        if char_vocab is None and dirname is None:
            raise Exception("Please provide either a directory with a valid PaReNT model or a list of characters to serve as the vocabulary for the model's character representation module.")

        elif char_vocab is not None and dirname is not None:
            raise Exception("Please provide EITHER a directory with a valid PaReNT model OR a list of characters to serve as the vocabulary for the model's character representation module.")

        elif char_vocab is None:
            char_vocab = pd.read_csv(f"./model/{dirname}/vocab.lst", header=0, na_filter=False, skip_blank_lines=False).squeeze(
            "columns").tolist()
            model_variables = parse_out_important(dirname)

            batch_sz = model_variables["batch_sz"]
            units = model_variables["units"]
            attention_units = model_variables["attention_units"]
            char_embedding_dim = model_variables["char_embedding_dim"]
            transblocks = model_variables["transblocks"]
            dropout = model_variables["dropout"]
            recurrent_dropout = model_variables["recurrent_dropout"]

        # Attributes
        self.char_vocab = char_vocab
        self.batch_sz = batch_sz
        self.embedding_model = embedding_model
        self.train_len = train_len
        self.attention_units = attention_units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.char_embedding_dim = char_embedding_dim
        self.units = units
        self.max_length_input = max_length_input
        self.max_length_output = max_length_output
        self.lr = lr
        self.transblocks = transblocks
        self.activation = activation
        self.optimizer = optimizer
        self.neutralize_chars = neutralize_chars
        self.neutralize_semant = neutralize_semant
        self.use_lang_embeddings = use_lang_embeddings
        self.semantic_warmup = semantic_warmup
        print(self.char_embedding_dim)


        # Aux layers
        self.char_list = [i[0] for i in Counter(self.char_vocab).most_common()]
        self.lookup_chars = tf.keras.layers.StringLookup(name="lookup_chars",
                                                         vocabulary=self.char_list,
                                                         mask_token="<pad>")
        self.char_embedding = tf.keras.layers.Embedding(input_dim=self.lookup_chars.vocabulary_size() + 1,
                                                        output_dim=self.char_embedding_dim,
                                                        name="char_embed")
        self.id_to_char = tf.keras.layers.StringLookup(name="id_to_char",
                                                       vocabulary=self.char_list,
                                                       mask_token="<pad>",
                                                       invert=True)

        # Modules
        self.lang_embedding = LanguageEmbedding()
        self.encoder = Encoder(enc_units=self.units,
                               attention_units=self.attention_units,
                               lookup_chars=self.lookup_chars,
                               char_embedding=self.char_embedding,
                               batch_sz=self.batch_sz,
                               activation=activation,
                               transblocks=self.transblocks,
                               recurrent_dropout=recurrent_dropout,
                               dropout=dropout)
        # self.decoder = Decoder(units=units, lookup_chars=self.lookup_chars, char_embedding=self.char_embedding,
        #                       id_to_char=self.id_to_char, attention_units=128)
        self.classifier = Classifier(fc_size=self.units,
                                     attention_units=self.attention_units,
                                     activation=activation,
                                     dropout=dropout)

        self.decoder = Decoder(char_embedding=self.char_embedding,
                               batch_sz=self.batch_sz,
                               dec_units=self.units,
                               vocab_size=len(self.char_list),
                               attention_type="luong",
                               max_length_input=self.max_length_input,
                               max_length_output=self.max_length_output,
                               activation=activation)

        # Optimizers and trackers

        self.steps_per_epoch = self.train_len // self.batch_sz
        self.clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=self.lr/10,
                                                      maximal_learning_rate=3*self.lr,
                                                      scale_fn=lambda x: 1 / (2. ** (x - 1)),
                                                      step_size=2 * self.steps_per_epoch
                                                      )
        self.optimizer = eval("tf.keras.optimizers." + optimizer)(learning_rate=self.clr,
                                                                  global_clipnorm=5.0)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        if dirname is not None:
            self.init_model()
            print(self.char_embedding_dim)
            self.load_weights(f"./model/{dirname}/model_weights.tf")

    def get_config(self): ## 5
        config = super(PaReNT, self).get_config()

        config["char_vocab"] = self.char_vocab
        config["batch_sz"] = self.batch_sz
        config["embedding_model"] = self.embedding_model
        config["train_len"] = self.train_len
        config["attention_units"] = self.attention_units
        config["dropout"]= self.dropout
        config["recurrent_dropout"] = self.recurrent_dropout
        config["char_embedding_dim"] = self.char_embedding_dim
        config["units"] = self.units
        config["max_length_input"] = self.max_length_input
        config["max_length_output"] = self.max_length_output
        config["lr"] = self.lr
        config["transblocks"] = self.transblocks
        config["activation"] = self.activation
        config["optimizer"] = self.optimizer
        config["neutralize_chars"] = self.neutralize_chars
        config["neutralize_semant"] = self.neutralize_semant
        config["use_lang_embeddings"] = self.use_lang_embeddings
        config["semantic_warmup"] = self.semantic_warmup

        return config


    def loss_function_seq(self, real, pred):
        # real shape = (BATCH_SIZE, max_length_output)
        # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        loss = cross_entropy(y_true=real, y_pred=pred)

        # Zero loss for padding
        mask = tf.logical_not(tf.math.equal(real, 0))  # output 0 for y=0 else output 1
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = mask * loss

        loss = tf.reduce_mean(loss)
        return loss

    def loss_function_class(self, real, pred, n_classes=3):
        cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False,
                                                                      reduction='none')
        real = tf.one_hot(tf.cast(real, tf.int32), n_classes)
        loss = cross_entropy(y_true=real, y_pred=pred)
        loss = tf.reduce_mean(loss)
        return loss

    @tf.function(reduce_retracing=True)
    def call(self, inputs,
             dec_input=None,
             *args, **kwargs):
        encoder_output, lang_embeddings = self.encode_hidden(inputs)

        enc_output, enc_h, enc_c = encoder_output

        # Set the AttentionMechanism object with encoder_outputs
        self.decoder.attention_mechanism.setup_memory(enc_output)

        # Create AttentionWrapperState as initial_state for decoder
        BATCH_SIZE = tf.cast(self.batch_sz, tf.int32)
        decoder_initial_state = self.decoder.build_initial_state(BATCH_SIZE,
                                                                 [enc_h, enc_c], tf.float32)
        retrieve_probs = self.decoder(dec_input, decoder_initial_state, lang_embeddings)
        classify_probs = self._classify(encoder_output, lang_embeddings)

        return retrieve_probs, classify_probs


    def encode_hidden(self, inputs) -> Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
        lang, char_inputs, semant_inputs = inputs
        self.set_batch_sz(tf.shape(lang)[0])

        lang_embeddings = self.lang_embedding(lang)
        # semant_inputs = self.semant_reshape(semant_inputs)

        hidden = self.encoder.initialize_hidden_state()
        encoded, enc_c, enc_h = self.encoder(char_inputs, semant_inputs,
                                             hidden=hidden,
                                             lang_embeddings=lang_embeddings,
                                             use_lang_embeddings=self.use_lang_embeddings,
                                             neutralize_chars=self.neutralize_chars,
                                             neutralize_semant=self.neutralize_semant)

        return (encoded, enc_c, enc_h), lang_embeddings


    @tf.function(reduce_retracing=True)
    def _classify(self, encoder_output: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
                  lang_embeddings: tf.Tensor) -> tf.Tensor:
        enc_out, enc_h, enc_c = encoder_output
        probs = self.classifier(enc_out, lang_embeddings)

        return probs

    def classify(self, list_of_lexemes: TupleList,
                 threshold=32):
        classified_lst =[]
        length = len(list_of_lexemes)

        for chunk in tqdm(chunker(list_of_lexemes, size=threshold), total=length // threshold, colour='#004C64'):
            inputs = self.preprocess(chunk)
            self.set_batch_sz(len(chunk))
            encoded, lang_embeddings = self.encode_hidden(inputs)
            probs = self._classify(encoded, lang_embeddings)

            del encoded
            del lang_embeddings

            classified_lst += list(tf.argmax(probs, axis=1).numpy())

        return classified_lst


    def preprocess(self, list_of_lexemes: TupleList) -> Tuple[tf.Tensor, tf.RaggedTensor, tf.RaggedTensor]:
        """
        :param list_of_lexemes: List of (language_code, lexeme) pairs.
        :return: Triple: (Tensor of size [batch_size, 1] representing language codes,
                          Ragged tensor of size [batch_size, None, None] containing character representation of lexeme,
                          Padded tensor of size [batch_size, 65, 300] containing semantic representations of lexeme)

        Note: Batch size must be statically known. If you get a shape mismatch error somewhere from the
        decoder/retriever, either set it manually or implicitly by calling this method on a list of lexemes
         of the desired length.
        """
        self.set_batch_sz(len(list_of_lexemes))

        lang_lst = []
        char_lst = []
        semant_lst = []

        for lang, lemma in list_of_lexemes:
            lang_encoded, char, semant = PaReNT_encode_tf(lang_code=lang, lexeme=lemma, model=self.embedding_model)
            lang_lst.append(lang_encoded), char_lst.append(char), semant_lst.append(semant)

        langs_tensor = tf.reshape(tf.convert_to_tensor(lang_lst), [-1, 1])
        char_tensor = tf.ragged.constant(char_lst)
        semant_tensor = tf.keras.preprocessing.sequence.pad_sequences(semant_lst,
                                                                      padding='post',
                                                                      maxlen=65,
                                                                      value=0.,
                                                                      dtype=float
                                                                      )
        return langs_tensor, char_tensor, semant_tensor

    def retrieve(self, list_of_lexemes: TupleList,
                 beam_size: int = 6,
                 decoding: Literal["greedy", "beam_search"] = "greedy",
                 threshold: int = 64) -> StringVec:
        retrieved_lst = []
        for chunk in tqdm(chunker(list_of_lexemes, size=threshold), total=len(list_of_lexemes) // threshold,
                          desc="Retrieving parent lemmas...", colour='#e6af00'):

            #self.set_batch_sz(len(chunk))
            inputs = self.preprocess(chunk)
            encoded, lang_embeddings = self.encode_hidden(inputs)

            probs = self._retrieve(encoded)
            argmaxed = tf.argmax(probs, axis=-1)
            chars = self.id_to_char(argmaxed)
            reduced = tf.strings.reduce_join(chars, axis=1).numpy()

            decoded = []
            for pair, retrieved_parents in zip(chunk, reduced):
                lang = pair[0]
                retrieved_parents = retrieved_parents.decode("UTF-8").split("[END]")[0].replace("_", " ")

                if lang == "ru":
                    decoded.append(latin_to_cyril(retrieved_parents))
                else:
                    decoded.append(retrieved_parents)

            for obj in [reduced, chars, argmaxed, probs]:
                del obj

            retrieved_lst += decoded

        return retrieved_lst

    def retrieve_and_classify(self,
                              list_of_lexemes,
                              threshold=64,
                              return_probs=False,
                              try_candidates=False,
                              num_candidates=6):

        retrieved_lst = []
        retrieve_probs_lst = []

        classified_lst = []
        classify_probs_lst = []

        candidates_lst = []

        del_padding = False

        if threshold > 1:
            chunks = tqdm(chunker(list_of_lexemes, size=threshold), total=int(np.ceil(len(list_of_lexemes) / threshold)),
                          desc="Retrieving parent lemmas and classifying...", colour='#e6af00')
        else:
            chunks = chunker(list_of_lexemes, size=threshold)

        for chunk in chunks:
            if len(chunk) < threshold:
                padding_len = threshold - len(chunk)
                chunk = chunk + [("en", "padding")]*padding_len
                del_padding = True

            inputs = self.preprocess(chunk)
            encoded, lang_embeddings = self.encode_hidden(inputs)
            probs = self._retrieve(encoded)
            retrieve_probs_lst += probs.numpy().tolist()

            argmaxed = tf.argmax(probs, axis=-1)
            chars = self.id_to_char(argmaxed)
            reduced = tf.strings.reduce_join(chars, axis=1).numpy()

            decoded = []
            for pair, retrieved_parents in zip(chunk, reduced):
                lang = pair[0]
                retrieved_parents = retrieved_parents.decode("UTF-8").split("[END]")[0].replace("_", " ")

                if lang == "ru":
                    decoded.append(latin_to_cyril(retrieved_parents))
                else:
                    decoded.append(retrieved_parents)

            for obj in [reduced, chars, argmaxed, probs]:
                del obj

            retrieved_lst += decoded

            probs = self._classify(encoded, lang_embeddings)
            classify_probs_lst += probs.numpy().tolist()
            classified_lst += list(tf.argmax(probs, axis=1).numpy())

            candidate_ids = self._retrieve_candidates(encoder_output=encoded, beam_width=num_candidates)
            candidates = self.id_to_char(candidate_ids)
            candidates_lst += [[y.decode("UTF-8").split("[END]")[0].replace("_", " ") for y in i] for i in tf.strings.reduce_join(candidates, axis=2).numpy().tolist()]

        if del_padding:
            retrieved_lst = retrieved_lst[:-padding_len]
            classified_lst = classified_lst[:-padding_len]
            candidates_lst = candidates_lst[:-padding_len]
            classify_probs_lst = classify_probs_lst[:-padding_len]
            retrieve_probs_lst = retrieve_probs_lst[:-padding_len]

        for i, double in enumerate(zip(candidates_lst, list_of_lexemes)):
            candlst, lang_and_lexeme = double
            lang, lexeme = lang_and_lexeme

            if lang == "ru":
                candidates_lst[i] = [latin_to_cyril(y) for y in candlst]

        if return_probs:
            if try_candidates:
                return retrieved_lst, classified_lst, retrieve_probs_lst, classify_probs_lst, candidates_lst
            else:
                return retrieved_lst, classified_lst, retrieve_probs_lst, classify_probs_lst
        else:
            if try_candidates:
                return retrieved_lst, classified_lst, candidates_lst
            else:
                return retrieved_lst, classified_lst


    def retrieve_whileloop(self, list_of_lexemes: TupleList,
                       threshold: int=256) -> StringVec:

        length = len(list_of_lexemes)

        if length <= threshold:
            return self._while_retrieve(list_of_lexemes, threshold=threshold)

        else:
            retrieved_lst = []
            for chunk in tqdm(chunker(list_of_lexemes, size=threshold), total=length//threshold):
                retrieved_lst += self.retrieve_whileloop(chunk, threshold=threshold)
            return retrieved_lst

    @tf.function(reduce_retracing=True)
    def _retrieve(self, encoder_output: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        enc_out, enc_h, enc_c = encoder_output

        start_tokens = tf.fill([self.decoder.batch_sz],
                               self.lookup_chars(b"[START]"))
        start_tokens = tf.cast(start_tokens, tf.int32)
        end_token = self.lookup_chars(b"[END]")
        end_token = tf.cast(end_token, tf.int32)


        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler(self.char_embedding)
        decoder_instance = tfa.seq2seq.BasicDecoder(cell=self.decoder.rnn_cell,
                                                    sampler=greedy_sampler,
                                                    output_layer=self.decoder.fc,
                                                    maximum_iterations=self.max_length_output)

        self.decoder.attention_mechanism.setup_memory(enc_out)
        decoder_initial_state = self.decoder.build_initial_state(self.batch_sz,
                                                                 [enc_h, enc_c],
                                                                 tf.float32)

        #decoder_embedding_matrix = self.decoder.char_embedding.variables[0]

        outputs, _, _ = decoder_instance(None,
                                         start_tokens=start_tokens,
                                         end_token=end_token,
                                         initial_state=decoder_initial_state)

        outputs = outputs.rnn_output

        # lang_embeddings = tf.expand_dims(lang_embeddings, axis=1)
        # correct_shape = [tf.shape(outputs)[0], tf.shape(outputs)[1], 2]
        # lang_embeddings = tf.broadcast_to(lang_embeddings, correct_shape)
        #
        # concatenated = self.decoder.concat([outputs, lang_embeddings])
        #probs = self.decoder.final(outputs)

        return outputs#probs


    def _retrieve_candidates(self,
                            encoder_output: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
                            beam_width=12) -> Tuple[tf.Tensor, tf.Tensor]:

        enc_out, enc_h, enc_c = encoder_output

        start_tokens = tf.fill([self.decoder.batch_sz],
                               self.lookup_chars(b"[START]"))
        start_tokens = tf.cast(start_tokens, tf.int32)
        end_token = self.lookup_chars(b"[END]")
        end_token = tf.cast(end_token, tf.int32)


        enc_out = tfa.seq2seq.tile_batch(enc_out, multiplier=beam_width)
        self.decoder.attention_mechanism.setup_memory(enc_out)
    #
        hidden_state = tfa.seq2seq.tile_batch([enc_h, enc_c], multiplier=beam_width)

        decoder_initial_state = self.decoder.build_initial_state(self.batch_sz * beam_width,
                                                                 hidden_state,
                                                                 tf.float32)

        decoder_initial_state = decoder_initial_state.clone(cell_state=hidden_state)

        decoder_instance = tfa.seq2seq.BeamSearchDecoder(self.decoder.rnn_cell,
                                                         beam_width=beam_width,
                                                         output_layer=self.decoder.fc,
                                                         output_all_scores=True,
                                                         length_penalty_weight=0.0001,
                                                         coverage_penalty_weight=0.00001,
                                                         maximum_iterations=100)
        decoder_embedding_matrix = self.decoder.char_embedding.variables[0]

        outputs, final_state, sequence_lengths = decoder_instance(decoder_embedding_matrix,
                                                                  start_tokens=start_tokens,
                                                                  end_token=end_token,
                                                                  initial_state=decoder_initial_state)
        #outputs = outputs.beam_search_decoder_output.scores

        # Convert the shape of outputs and beam_scores to (inference_batch_size, beam_width, time_step_outputs)
        final_outputs = tf.transpose(outputs.predicted_ids, perm=(0, 2, 1))
        #beam_scores = tf.transpose(outputs.beam_search_decoder_output.scores, perm=(0, 2, 1))

        return final_outputs#, beam_scores

    def set_batch_sz(self, batch_sz):
        self.batch_sz, self.encoder.batch_sz, self.decoder.batch_sz = (batch_sz, batch_sz, batch_sz)

    def plot_lang_embeddings(self):
        langs = ["en", "de", "nl", "es", "fr", "cs", "ru"]
        embeddings = self.lang_embedding(langs)

        df_embeddings = pd.DataFrame()
        df_embeddings["language"] = langs
        df_embeddings["x"] = embeddings[:, 0]
        df_embeddings["y"] = embeddings[:, 1]

        plot = (so.Plot(df_embeddings,
                        x="x",
                        y="y",
                        text="language")
                .add(so.Dot())
                .add(so.Text(valign="bottom"))
                )

        return plot

    def dendrogram_lang_embeddings(self, path):
        langs = ["en", "de", "nl", "es", "fr", "cs", "ru"]
        embeddings = self.lang_embedding(langs)

        df_embeddings = pd.DataFrame()
        df_embeddings["x"] = embeddings[:, 0]
        df_embeddings["y"] = embeddings[:, 1]

        linked = h.linkage(df_embeddings)
        h.dendrogram(linked, labels=langs)
        plt.savefig(path)

    @tf.function(reduce_retracing=True)
    def train_step(self, data, epoch: int):
        if epoch < self.semantic_warmup:
            self.neutralize_chars = True
        else:
            self.neutralize_chars = False

        inp, targ = data
        targ_seq, targ_class = targ

        with tf.GradientTape() as tape:
            dec_input = self.lookup_chars(targ_seq)

            dec_input = dec_input[:, :-1]  # Ignore <end> token

            real_seq = self.lookup_chars(targ_seq)
            real_seq = real_seq[:, 1:]  # ignore <start> token
            real_seq = real_seq.to_tensor()


            retrieve_probs, classify_probs = self(inp,
                                                   training=True,
                                                   dec_input=dec_input)

            loss_seq = self.loss_function_seq(real_seq, retrieve_probs)
            loss_class = self.loss_function_class(targ_class, classify_probs)
            loss = loss_seq + loss_class*0.01 #May need 0.0001*loss_class

        variables = self.encoder.variables + self.decoder.variables + self.lang_embedding.variables + self.classifier.variables
        gradients = tape.gradient(loss, variables)
        #self.optimizer_seq.apply_gradients(zip(gradients, variables))
        self.optimizer.apply_gradients(zip(gradients, variables))
        #
        #
        # with tf.GradientTape(persistent=True) as tape:
        #     _, classify_probs = self(inp,
        #                             training=True,
        #                             dec_input=dec_input)
        #
        #     loss_class = self.loss_function_class(targ_class, classify_probs)
        #
        # variables = self.encoder.trainable_variables + \
        #             self.lang_embedding.trainable_variables + \
        #             self.classifier.trainable_variables


        # gradients = tape.gradient(loss_class, variables)
        # self.optimizer_class.apply_gradients(zip(gradients, variables))

        return loss_seq, loss_class

    def train(self, train_data: pd.DataFrame,
              derinet,
              batch_sz: int = 32,
              epochs: int = 2,
              test_mode: bool = False,
              frac_mode: bool = False):

        for epoch in range(epochs):
            start = time.time()

            if test_mode and (epoch == 0):
                train_data = train_data.sample(frac=0.01)

            train_generator = data_generator(train_data,
                                             multibpe_model,
                                             batch_size=batch_sz)

            self.set_batch_sz(batch_sz)
            training_size = len(train_data)
            plots = []

            if training_size is not None:
                total = training_size // self.batch_sz
                iterator = tqdm(enumerate(train_generator), total=total)
            else:
                iterator = tqdm(enumerate(train_generator))

            total_loss_seq = 0
            total_loss_class = 0
            # print(enc_hidden[0].shape, enc_hidden[1].shape
            for (batch, data) in iterator:
                self.set_batch_sz(data[0][0].shape[0])
                seq_loss, class_loss = self.train_step(data, epoch=epoch)
                total_loss_seq += seq_loss
                total_loss_class += class_loss

                if batch % 1000 == 0:
                    tqdm.write(
                        f'Epoch {epoch + 1} Batch {batch,} Seq Loss {seq_loss.numpy()} Class loss {class_loss.numpy()}' + "\n")
                    tqdm.write("Classification sanity check:" + "\n")
                    tqdm.write(str(self.classify(sanity)) + "\n")
                    check = self._while_retrieve(sanity, threshold=64)
                    tqdm.write("Retrieval sanity check:" + "\n")
                    tqdm.write(str([*zip(sanity, check)])+ "\n")
                    tqdm.write(str(self._while_retrieve([("cs", "láska"), ("en", "love")], threshold=64)) + "\n")

                #if batch % 100 == 0:
                    #plots.append(self.plot_lang_embeddings())


            print(f'Epoch {epoch + 1} Seq Loss {total_loss_seq} Class loss {total_loss_class}')
            print(f'Time taken for 1 epoch {time.time() - start} sec\n')

            tf.print(f'Epoch {epoch + 1} Seq Loss {total_loss_seq} Class loss {total_loss_class}')
            tf.print(f'Time taken for 1 epoch {time.time() - start} sec\n')

            if epoch % 2 == 0:
                self.evaluate(subset="test",
                              derinet=derinet,
                              epoch=epoch+1,
                              frac_mode=frac_mode,
                              cluster=args.cluster)

    def evaluate(self,
                 derinet,
                 subset="test",
                 epoch: int = 0,
                 frac_mode = True,
                 cluster = False,
                 threshold = 64):
        print("EVALUATING!")

        if cluster:
            try:
                path = f"../tf_models/" + f"e{epoch}-" + '-'.join(f'{k[0:3]}={v}' for k, v in vars(args).items())
            except:
                path = f"./PaReNT_final_evaluation/"

            if not os.path.exists(path):
                os.makedirs(path)

            self.save_weights(path + "/model_weights.tf")

            h5_dir = path + "/h5_weights/"
            if not os.path.exists(h5_dir):
                os.mkdir(h5_dir)
            self.save_weights(h5_dir + "model.weights.h5")

        else:
            path = f"./PaReNT_final_evaluation/"

        if not os.path.exists(path):
            os.makedirs(path)

        test_data = load_df(subset=subset,
                            cluster=cluster)

        if frac_mode:
            test_data = test_data.sample(frac=0.01, random_state=42)

        inp = [*zip(test_data["language"], test_data["lexeme"])]
        retrievals, class_predictions = self.retrieve_and_classify(inp, threshold=threshold)
        classifier_dict = {"Unmotivated": 0, "Derivative": 1, "Compound": 2}
        classifier_dict_inv = {i: k for i, k in zip(classifier_dict.values(), classifier_dict.keys())}
        ground_truth = [classifier_dict[i] for i in test_data["word_type"]]
        print("Class Bal acc overall:", balanced_accuracy_score(ground_truth, class_predictions))
        test_data["PaReNT_classify"] = [classifier_dict_inv[i] for i in class_predictions]

        print(2)

        test_data["PaReNT_retrieve"] = retrievals

        gb = test_data.groupby("language")
        dfs_by_language = [gb.get_group(group) for group in gb.groups.keys()]

        table = pt.PrettyTable()
        table.field_names = ["Language", "Retrieval accuracy", "Classification accuracy",
                             "Balanced classification accuracy"]

        classify_lst = []
        nonbal_classify_lst = []
        retrieve_lst = []
        print(3)

        for df_lang in dfs_by_language:
            classification_acc = balanced_accuracy_score(df_lang["word_type"], df_lang["PaReNT_classify"])
            classification_acc_nonbal = accuracy_score(df_lang["word_type"], df_lang["PaReNT_classify"])
            retrieval_acc = retrieval_accuracy(df_lang["parents"], df_lang["PaReNT_retrieve"])
            print(retrieval_acc)

            classify_lst.append(classification_acc)
            nonbal_classify_lst.append(classification_acc_nonbal)
            retrieve_lst.append(retrieval_acc)

            table.add_row([list(df_lang.language)[0], retrieval_acc, round(classification_acc_nonbal, 2),
                           round(classification_acc, 2)])


        test_data.to_csv(path + "/for_analysis.tsv", sep="\t",
                         index_label=False, index=False)

        table.add_row(
            ["Total", round(np.mean(retrieve_lst), 2), round(np.mean(nonbal_classify_lst), 2),
             round(np.mean(classify_lst), 2)])

        with open(path + "/table.txt", "w+") as file:
            file.write(table.get_string())

        plot = self.plot_lang_embeddings()
        plot.save(path + "/plot.pdf")

        vocab = pd.DataFrame()
        vocab["1"] = self.char_list
        vocab.to_csv(path + "/vocab.lst", sep="\t",
               index_label=False, index=False)

        df = pd.read_csv(path + "/for_analysis.tsv", sep="\t")

        df = df[df.language == "cs"]

        try:
            tree_acc = tree_accuracy(list(df.PaReNT_retrieve), list(df.parents), derinet)

            with open(path + "/tree_acc.txt", "w") as file:
                file.write(str(tree_acc))
        except:
            pass

        print("Done!")
        print(table)

    def init_model(self):
        example = pd.DataFrame(
            {"lexeme": ["a"], "parents": ["a"], "language": ["cs"], "word_type": ["Unmotivated"], "block": ["a"],
             "dataset_name": [""]})
        example = preprocess_data(example, start=0., end=1., model=self.embedding_model)

        inp, targ = example[0]
        targ_seq, targ_class = targ
        dec_input = self.lookup_chars(targ_seq)
        dec_input = dec_input[:, :-1]  # Ignore <end> token

        retrieve_probs, classify_probs = self(inp,
                                               training=True,
                                               dec_input=dec_input)

        print(f"Initialization sanity check: {retrieve_probs}, {classify_probs}")

    def _cond(self, lexemes, i, outlst):
        length = len(lexemes)
        return tf.less(i, length)

    def _body(self, lexemes, i, outlst):
        outlst[i] = self.retrieve([lexemes[i]], decoding="greedy")[0]
        i += 1
        return lexemes, i, outlst

    def _while_retrieve(self, lexemes, threshold):
        i = 0
        outlst = list(np.repeat("0", len(lexemes)))

        out = tf.while_loop(cond=self._cond,
                            body=self._body,
                            loop_vars=[lexemes, i, outlst],
                            parallel_iterations=threshold,
                            swap_memory=True)[-1]
        out = [i.numpy().decode("UTF-8").split("[END]")[0].replace("_", " ") for i in out]

        return out

    def _classify_body(self, lexemes, i, outlst):
        outlst[i] = self.classify([lexemes[i]])[0]
        i += 1
        return lexemes, i, outlst

    def _while_classify(self, lexemes):
        i = 0
        outlst = list(np.repeat(0, len(lexemes)))

        out = tf.while_loop(cond=self._cond,
                            body=self._classify_body,
                            loop_vars=[lexemes, i, outlst],
                            parallel_iterations=100,
                            swap_memory=True
                            )[-1]
        return [i.numpy() for i in out]

#
# def inp(x, dtype=tf.float32):
#     return tf.keras.layers.Input(x, dtype=dtype)
#
# inps = [*train_data.take(1).batch(1)][0]
#
#
# model = PaReNT(for_vocab, batch_sz=1)
#
# ##


if __name__ == "__main__":
    try:
        import scripts.derinet
        from scripts.derinet.lexicon import Lexicon
    except:
        import derinet
        from derinet.lexicon import Lexicon

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch_name', type=str, help='Architecture name')
    parser.add_argument('--cluster', type=bool, default=True, help='Cluster or home?')
    parser.add_argument('--batch_sz', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='No. of epochs')
    parser.add_argument('--units', type=int, default=512, help='No. of units in most layers')
    parser.add_argument('--attention_units', type=int, default=64, help='No. of units in attention layers')
    parser.add_argument('--char_embeddings', type=int, default=64, help='Dim of character embeddings')
    parser.add_argument('--test_mode', type=int, help='Use only one hundredth of the available data')
    parser.add_argument('--transblocks', type=int, default=0, help='Number of transformer-like blocks in the encoder')
    parser.add_argument("--length_penalty", type=float, default=0.0, help='Penalty of the decoder ')
    parser.add_argument("--frac_mode", type=int, default=1, help='Evaluate only on 1% of the data')

    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default="Adam", help='Optimizer - Adam or Nadam')
    parser.add_argument('--dropout', type=float, default=0.1, help='Fully-connected dropout rate')
    parser.add_argument('--recurrent_dropout', type=float, default=0.0, help='Recurrent dropout rate')
    parser.add_argument('--l', type=str, default="none", help='L-regularization {none (default), l1, l2, l1_l2}')

    parser.add_argument('--use_lang_embeddings', type=int, default=1, help='Dropout rate')
    parser.add_argument('--neutralize_chars', type=int, default=0, help='Multiply all character vectors by zero')
    parser.add_argument('--neutralize_semant', type=int, default=0, help='Multiply all semantic vectors by zero')

    parser.add_argument('--semantic_warmup', type=int, default=0,
                        help='Number of epochs after which training on char vectors kicks in')

    args = parser.parse_args()

    batch_sz = args.batch_sz

    print(sys.argv)
    print(args)


    def tree_accuracy(model_output: StringVec, ground_truth: StringVec, data_source: derinet.lexicon.Lexicon) -> float:
        assert len(model_output) == len(ground_truth)
        print("Calculating tree acc")

        acc_lst = []
        for x, Y in tqdm(zip(model_output, ground_truth), total=len(ground_truth), desc="Calculating tree accuracy:"):
            if x == Y:
                acc_lst.append(True)
            else:
                x = x.split(" ")
                Y = Y.split(" ")

                x_lexemes_lst = [data_source.get_lexemes(i) for i in x]
                Y_lexemes_lst = [data_source.get_lexemes(i) for i in Y]

                if (not (contains_empty_lst(x_lexemes_lst) or contains_empty_lst(Y_lexemes_lst))) and (
                        len(x_lexemes_lst) == len(Y_lexemes_lst)):
                    inner_lst = []

                    for x_lexemes, Y_lexemes in zip(x_lexemes_lst, Y_lexemes_lst):
                        inner_lst.append(
                            any([x_lexeme.get_tree_root().lemid == Y_lexeme.get_tree_root().lemid for x_lexeme, Y_lexeme
                                 in
                                 product(x_lexemes, Y_lexemes)]))

                    acc_lst.append(all(inner_lst))
                else:
                    acc_lst.append(False)

        return sum(acc_lst) / len(acc_lst)

    if args.test_mode:
        # args.test_mode = False
        print("RUNNING IN TEST MODE")
    else:
        print("RUNNING IN REAL MODE")

    lexicon = Lexicon()
    try:
        lexicon.load("./data_raw/Czech/derinet-2-1.tsv", on_err='continue')
    except:
        lexicon.load("../data_raw/Czech/derinet-2-1.tsv", on_err='continue')

    train_df = load_df(subset="train",
                       cluster=args.cluster)
    print(train_df)
    for_vocab = get_vocab(train_df)

    model = PaReNT(for_vocab,
                   train_len=len(train_df),
                   embedding_model=multibpe_model,
                   batch_sz=args.batch_sz,
                   units=args.units,
                   attention_units=args.attention_units,
                   char_embedding_dim=args.char_embeddings,
                   lr=args.lr,
                   transblocks=args.transblocks,
                   activation=tf.nn.swish,
                   recurrent_dropout=args.recurrent_dropout,
                   dropout=args.dropout,
                   optimizer=args.optimizer,
                   use_lang_embeddings=bool(args.use_lang_embeddings),
                   neutralize_chars=bool(args.neutralize_chars),
                   neutralize_semant=bool(args.neutralize_semant),
                   semantic_warmup=args.semantic_warmup)

    sanity = [('ru', "человек"), ('cs', 'centnýřský'), ('en', 'seven-hearted'), ('cs', 'pes'), ('en', 'dog'),
              ('en', 'blackheart'), ('cs', 'černoušek'), ('cs', 'černoušec')]

    frac_mode = bool(args.frac_mode)

    print(model.lr)
    model.train(train_df,
                derinet=lexicon,
                epochs=args.epochs,
                batch_sz=args.batch_sz,
                test_mode=args.test_mode,
                frac_mode=frac_mode)
