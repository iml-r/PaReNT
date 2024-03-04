# d231a45b-25a9-11ec-986f-f39926f24a9c
# 1ca06d3c-25bf-11ec-986f-f39926f24a9c
# dc37826b-25a8-11ec-986f-f39926f24a9c

import argparse
import datetime
import os
import re
from tqdm import tqdm

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from transformers import AutoTokenizer

from morpho_dataset import MorphoDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=40, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--cle_dim", default=256, type=int, help="CLE embedding dimension.")
parser.add_argument("--rnn_dim", default=512, type=int, help="RNN cell dimension.")

parser.add_argument("--dropout", default=0.5, type=float, help="Dropout.")
parser.add_argument("--vec_dropout", default=0.2, type=float, help="Embeddings dropout.")
parser.add_argument("--layers", default=3, type=int, help="Layers.")

parser.add_argument("--char_masking_rate", default=0, type=float, help="Char masking rate.")

args = parser.parse_args([] if "__file__" not in globals() else None)

# Create logdir name
args.logdir = os.path.join("logs", "{}-{}-{}".format(
    os.path.basename(globals().get("__file__", "notebook")),
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
))

#args.path = Path.home() / 'lemmatizer'
args.path = ""

# Fix random seeds and threads
tf.keras.utils.set_random_seed(args.seed)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)

# def helper functions
def get_vocab(tensor: tf.RaggedTensor) -> list:
    tensor = tf.reshape(tensor, [-1])
    uniq = tf.unique(tensor)
    vocab = list(uniq[0].numpy())
    vocab.sort()

    vocab = ["[BOW]", "[EOW]"] + vocab
    return vocab


# def load_vecs(what):
#     filename = args.path / f'lemmatizer_{what}.npy'
#     with open(filename, 'rb') as f:
#         length = np.load(f)
#         length = length.item()
#         li = [np.load(f) for _ in range(length)]
#         li = [tf.constant(item) for item in li]
#         row_lengths = [len(item) for item in li]
#         conc = tf.concat(li, axis=0)
#         return tf.RaggedTensor.from_row_lengths(values=conc,
#                                                 row_lengths=row_lengths)
#
#
# train_vecs = load_vecs('train')
# dev_vecs = load_vecs('dev')
# test_vecs = load_vecs('test')

#print(train_vecs.shape, dev_vecs.shape, test_vecs.shape)

generator = tf.random.Generator.from_seed(args.seed)
tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

##Load data################
print("Loading training data...")
pdlist = []
file_path = "data/train/"
for i in tqdm(os.listdir(file_path)):
    df = pd.read_csv(file_path+i, sep=".")
    pdlist.append(df)
train = pd.concat(pdlist)
#del pdlist

train = train.sample(frac=1)
#FIXME:
train = train.dropna()

# train_x = list(funcy.partition(64, list(train['lexeme'])))
# train_Y = list(funcy.partition(64, list(train['parents'])))

##https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/t5#training
train_x = list(train['lexeme'])
train_x = tf.strings.bytes_split([i for i in train_x])
x_vocab = get_vocab(train_x)


train_Y = list(train['parents'])
train_Y = tf.strings.bytes_split([i for i in train_Y])
Y_vocab = get_vocab(train_Y)
#max_target_length = longest(train_Y)

#maxlen = np.max([max_source_length, max_target_length])

# print("Encoding...")
# encoding_x = tokenizer(
#     train_x,
#     padding='max_length',
#     max_length=maxlen,
#     truncation=True,
#     return_tensors="tf",
# )
#
# input_ids_x, attention_mask_x = encoding_x.input_ids, encoding_x.attention_mask
# input_ids_x, attention_mask_x = prepend_int_matrix(input_ids_x, 2), prepend_int_matrix(attention_mask_x, 2)
#
# encoding_Y = tokenizer(
#     train_Y,
#     padding='max_length',
#     max_length=maxlen,
#     truncation=True,
#     return_tensors="tf"
# )

# labels = encoding_Y.input_ids
# labels = prepend_int_matrix(labels, 2)
#
# decoder_ids = encoding_Y.input_ids
# decoder_ids = prepend_int_matrix(decoder_ids, 2)

class Encoder(tf.keras.layers.Layer):
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.lstms = []
        for _ in range(args.layers):
            rnn = tf.keras.layers.LSTM(args.rnn_dim, return_sequences=True)
            bidir = tf.keras.layers.Bidirectional(rnn, merge_mode="sum")
            self.lstms.append(bidir)
        self.dropouts = [tf.keras.layers.Dropout(rate=args.dropout) for _ in range(args.layers)]

    def call(self, input, mask):
        if self.args.char_masking_rate > 0:
            char_masking_shape = tf.concat((tf.shape(input)[:2], [1]), axis=0)
            char_masking_prob = generator.uniform(shape=char_masking_shape)
            char_masking_mult = tf.cast(char_masking_prob > self.args.char_masking_rate, tf.float32)
            input = input * char_masking_mult
        for i in range(self.args.layers):
            res = self.lstms[i](input, mask=mask)
            res = self.dropouts[i](res)
            input = res + input if i > 0 else res
        return input


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, scheduler, train_x, train_Y) -> None:
        super().__init__()

        self.args = args

        self.vocab_x = get_vocab(train_x)
        self.vocab_Y = get_vocab(train_Y)

        self.source_mapping = tf.keras.layers.StringLookup(vocabulary=self.vocab_x)
        self.target_mapping = tf.keras.layers.StringLookup(vocabulary=self.vocab_Y)
        self.target_mapping_inverse = type(self.target_mapping)(
            vocabulary=self.target_mapping.get_vocabulary(), invert=True)

        # TODO(lemmatizer_noattn): Define
        # - `self.source_embedding` as an embedding layer of source chars into `args.cle_dim` dimensions
        self.source_embedding = tf.keras.layers.Embedding(self.source_mapping.vocabulary_size(), args.cle_dim)

        # TODO: Define
        # - `self.source_rnn` as a bidirectional GRU with `args.rnn_dim` units, returning **whole sequences**,
        #   summing opposite directions
        # self.source_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(args.rnn_dim, return_sequences=True), merge_mode="sum")

        self.source_rnn = Encoder(args)

        # TODO(lemmatizer_noattn): Then define
        # - `self.target_embedding` as an embedding layer of target chars into `args.cle_dim` dimensions
        # - `self.target_rnn_cell` as a GRUCell with `args.rnn_dim` units
        # - `self.target_output_layer` as a Dense layer into as many outputs as there are unique target chars
        self.target_embedding = tf.keras.layers.Embedding(self.target_mapping.vocabulary_size(), args.cle_dim)
        self.target_rnn_cell = tf.keras.layers.GRUCell(args.rnn_dim)
        self.target_output_layer = tf.keras.layers.Dense(self.target_mapping.vocabulary_size())
        self.out_dim = self.target_mapping.vocabulary_size()

        # TODO: Define
        # - `self.attention_source_layer` as a Dense layer with `args.rnn_dim` outputs
        # - `self.attention_state_layer` as a Dense layer with `args.rnn_dim` outputs
        # - `self.attention_weight_layer` as a Dense layer with 1 output
        self.attention_source_layer = tf.keras.layers.Dense(args.rnn_dim)
        self.attention_state_layer = tf.keras.layers.Dense(args.rnn_dim)
        self.attention_weight_layer = tf.keras.layers.Dense(1)

        # Compile the model
        self.compile(
            optimizer=tf.optimizers.Adam(learning_rate=scheduler),
            loss=lambda y_true, y_pred: tf.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true.values,
                                                                                                  y_pred.values),
            metrics=[tf.metrics.Accuracy(name="accuracy")],
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    class DecoderTraining(tfa.seq2seq.BaseDecoder):
        def __init__(self, lemmatizer, *args, **kwargs):
            self.lemmatizer = lemmatizer
            super().__init__.__wrapped__(self, *args, **kwargs)

        @property
        def batch_size(self):
            # TODO(lemmatizer_noattn): Return the batch size of `self.source_states` as a *scalar* number;
            # use `tf.shape` to get the full shape and then extract the batch size.
            return tf.shape(self.source_states)[0]

        @property
        def output_size(self):
            # TODO(lemmatizer_noattn): Describe the size of a single decoder output (batch size and the
            # sequence length are not included) by returning
            #   tf.TensorShape(number of logits of each output element [lemma character])
            return tf.TensorShape([self.lemmatizer.out_dim])

        @property
        def output_dtype(self):
            # TODO(lemmatizer_noattn): Return the type of the decoder output (so the type of the
            # produced logits).
            return tf.float32

        def with_attention(self, inputs, states):
            # TODO: Compute the attention.
            # - Compute projected source states by passing `self.source_states` through the
            #   `self.lemmatizer.attention_source_layer`. Because `self.source_states` do not change,
            #   you should in fact precompute the projected source states once in `initialize`.
            # - Compute projected decoder state by passing `states` though `self.lemmatizer.attention_state_layer`.
            # - Sum the two projections. However, the first has shape [a, b, c] and the second [a, c]. Therefore,
            #   expand the second to [a, b, c] or [a, 1, c] (the latter works because of broadcasting rules).
            # - Pass the sum through `tf.tanh` and through the `self.lemmatizer.attention_weight_layer`.
            # - Then, run softmax on a suitable axis, generating `weights`.
            # - Multiply the original (non-projected) `self.source_states` with `weights` and sum the result
            #   in the axis corresponding to characters, generating `attention`. Therefore, `attention` is
            #   a fixed-size representation for every batch element, independently on how many characters
            #   the corresponding input forms had.
            # - Finally concatenate `inputs` and `attention` (in this order) and return the result.

            decoder_states = self.lemmatizer.attention_state_layer(states)
            summed_states = self.precomputed_source_states + tf.expand_dims(decoder_states, 1)

            comp_states = self.lemmatizer.attention_weight_layer(tf.tanh(summed_states))
            weights = tf.math.softmax(comp_states, axis=1)

            attention = tf.math.reduce_sum(weights * self.source_states, axis=1)

            return tf.concat([inputs, attention], axis=1)

        def initialize(self, layer_inputs, initial_state=None, mask=None):
            self.source_states, self.targets = layer_inputs

            # TODO(lemmatizer_noattn): Define `finished` as a vector of self.batch_size of `False` [see tf.fill].
            finished = tf.zeros((self.batch_size,), dtype=bool)

            # TODO(lemmatizer_noattn): Define `inputs` as a vector of self.batch_size of MorphoDataset.Factor.BOW,
            # embedded using self.lemmatizer.target_embedding
            inputs = tf.ones((self.batch_size,)) * MorphoDataset.Factor.BOW
            inputs = self.lemmatizer.target_embedding(inputs)

            # TODO: Define `states` as the representation of the first character
            # in `source_states`. The idea is that it is most relevant for generating
            # the first letter and contains all following characters via the backward RNN.
            states = self.source_states[:, 0, :]
            self.precomputed_source_states = self.lemmatizer.attention_source_layer(self.source_states)

            # TODO: Pass `inputs` through `self.with_attention(inputs, states)`.
            inputs = self.with_attention(inputs, states)

            return finished, inputs, states

        def step(self, time, inputs, states, training):
            # TODO(lemmatizer_noattn): Pass `inputs` and `[states]` through self.lemmatizer.target_rnn_cell,
            # which returns `(outputs, [states])`.
            outputs, [states] = self.lemmatizer.target_rnn_cell(inputs, [states])

            # TODO(lemmatizer_noattn): Overwrite `outputs` by passing them through self.lemmatizer.target_output_layer,
            outputs = self.lemmatizer.target_output_layer(outputs)

            # TODO(lemmatizer_noattn): Define `next_inputs` by embedding `time`-th chars from `self.targets`.
            next_inputs = self.lemmatizer.target_embedding(self.targets[:, time])

            # TODO(lemmatizer_noattn): Define `finished` as a vector of booleans; True if the corresponding
            # `time`-th char from `self.targets` is `MorphoDataset.Factor.EOW`, False otherwise.
            finished = self.targets[:, time] == MorphoDataset.Factor.EOW

            # TODO: Pass `next_inputs` through `self.with_attention(next_inputs, states)`.
            next_inputs = self.with_attention(next_inputs, states)

            return outputs, states, next_inputs, finished

    class DecoderPrediction(DecoderTraining):
        @property
        def output_size(self):
            # TODO(lemmatizer_noattn): Describe the size of a single decoder output (batch size and the
            # sequence length are not included) by returning a suitable
            # `tf.TensorShape` representing a *scalar* element, because we are producing
            # lemma character indices during prediction.
            return tf.TensorShape([])

        @property
        def output_dtype(self):
            # TODO(lemmatizer_noattn): Return the type of the decoder output (i.e., target lemma character indices).
            return tf.int32

        def initialize(self, layer_inputs, initial_state=None, mask=None):
            # Use `initialize` from the `DecoderTraining`, passing None as `targets`.
            return super().initialize([layer_inputs, None], initial_state)

        def step(self, time, inputs, states, training):
            # TODO(lemmatizer_noattn): Pass `inputs` and `[states]` through self.lemmatizer.target_rnn_cell,
            # which returns `(outputs, [states])`.
            outputs, [states] = self.lemmatizer.target_rnn_cell(inputs, [states])

            # TODO(lemmatizer_noattn): Overwrite `outputs` by passing them through self.lemmatizer.target_output_layer,
            outputs = self.lemmatizer.target_output_layer(outputs)

            # TODO(lemmatizer_noattn): Overwrite `outputs` by passing them through `tf.argmax` on suitable axis and with
            # `output_type=tf.int32` parameter.
            outputs = tf.argmax(outputs, axis=1, output_type=tf.int32)

            # TODO(lemmatizer_noattn): Define `next_inputs` by embedding the `outputs`
            next_inputs = self.lemmatizer.target_embedding(outputs)

            # TODO(lemmatizer_noattn): Define `finished` as a vector of booleans; True if the corresponding
            # prediction in `outputs` is `MorphoDataset.Factor.EOW`, False otherwise.
            finished = outputs == MorphoDataset.Factor.EOW

            # TODO(DecoderTraining): Pass `next_inputs` through `self.with_attention(next_inputs, states)`.
            next_inputs = self.with_attention(next_inputs, states)

            return outputs, states, next_inputs, finished

    # If `targets` is given, we are in the teacher forcing mode.
    # Otherwise, we run in autoregressive mode.
    def call(self, inputs, training=False, targets=None, *, vectors):
        # Forget about sentence boundaries and instead consider
        # all valid form-lemma pairs as independent batch examples.
        #
        # Then, split the given forms into character sequences and map then
        # to their indices.

        source_charseqs = inputs.values

        vectors_seqs = vectors.values

        source_charseqs = tf.strings.bytes_split(source_charseqs, "UTF-8")
        source_charseqs = self.source_mapping(source_charseqs)
        if targets is not None:
            # The targets are already mapped sequences of characters, so only
            # drop the sentence boundaries, and convert to a dense tensor
            # (the EOW correctly indicate end of lemma).
            target_charseqs = targets.values
            target_charseqs = target_charseqs.to_tensor()

        # TODO(lemmatizer_noattn): Embed source_charseqs using `source_embedding`
        embedded_ragged = self.source_embedding(source_charseqs)
        embedded_tensor = embedded_ragged.to_tensor()

        #vectors_tensor = vectors_seqs[:, None, :]

        #vectors_tensor = tf.broadcast_to(vectors_tensor, tf.shape(embedded_tensor))
        #vectors_tensor = tf.cast(vectors_tensor, tf.float32)

       # if self.args.vec_dropout > 0 and training:
       #     vectors_tensor = tf.nn.dropout(vectors_tensor, self.args.vec_dropout)

        #concat_tensor = tf.concat((embedded_tensor, vectors_tensor), axis=-1)

        # TODO: Run source_rnn on the embedded sequences, returning outputs in `source_states`.
        # However, convert the embedded sequences from a RaggedTensor to a dense Tensor first,
        # i.e., call the `source_rnn` with
        #   (source_embedded.to_tensor(), mask=tf.sequence_mask(source_embedded.row_lengths()))

        sources_states = self.source_rnn(embedded_tensor,
                                         mask=tf.sequence_mask(embedded_ragged.row_lengths()))

        # Run the appropriate decoder. Note that the outputs of the decoders
        # are exactly the outputs of `tfa.seq2seq.dynamic_decode`.
        if targets is not None:
            # TODO(lemmatizer_noattn): Create a self.DecoderTraining by passing `self` to its constructor.
            # Then run it on `[source_states, target_charseqs]` input,
            # storing the first result in `output` and the third result in `output_lens`.
            decoder_t = self.DecoderTraining(self)
            output, _, output_lens = decoder_t([sources_states, target_charseqs])
        else:
            # TODO(lemmatizer_noattn): Create a self.DecoderPrediction by using:
            # - `self` as first argument to its constructor
            # - `maximum_iterations=tf.cast(source_charseqs.bounding_shape(1) + 10, tf.int32)`
            #   as another argument, which indicates that the longest prediction
            #   must be at most 10 characters longer than the longest input.
            #
            # Then run it on `source_states`, storing the first result in `output`
            # and the third result in `output_lens`. Finally, because we do not want
            # to return the `[EOW]` symbols, decrease `output_lens` by one.
            maximum_iterations = tf.cast(source_charseqs.bounding_shape(1) + 10, tf.int32)
            decoder_p = self.DecoderPrediction(self, maximum_iterations=maximum_iterations)
            output, _, output_lens = decoder_p(sources_states)
            output_lens -= 1

        # Reshape the output to the original matrix of lemmas
        # and explicitly set mask for loss and metric computation.
        output = tf.RaggedTensor.from_tensor(output, output_lens)
        output = inputs.with_values(output)
        return output

    def train_step(self, data):
        x, y, vectors = data

        # Convert `y` by splitting characters, mapping characters to ids using
        # `self.target_mapping` and finally appending `MorphoDataset.Factor.EOW`
        # to every sequence.
        y_targets = self.target_mapping(tf.strings.bytes_split(y.values, "UTF-8"))
        y_targets = tf.concat(
            [y_targets, tf.fill([y_targets.bounding_shape(0), 1],
                                tf.constant(MorphoDataset.Factor.EOW, tf.int64))], axis=-1)
        y_targets = y.with_values(y_targets)

        with tf.GradientTape() as tape:
            y_pred = self(x, targets=y_targets, vectors=vectors, training=True)
            loss = self.compute_loss(x, y_targets.values, y_pred.values)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": metric.result() for metric in self.metrics if metric.name == "loss"}

    def predict_step(self, data):
        x, y, vectors = data
        y_pred = self(x, vectors=vectors, training=False)
        y_pred = self.target_mapping_inverse(y_pred)
        y_pred = tf.strings.reduce_join(y_pred, axis=-1)
        return y_pred

    def test_step(self, data):
        x, y, vectors = data
        y_pred = self.predict_step(data)
        self.compiled_metrics.update_state(
            tf.ones_like(y.values, dtype=tf.int32), tf.cast(y_pred.values == y.values, tf.int32))
        return {m.name: m.result() for m in self.metrics if m.name != "loss"}


# Load the data. Using analyses is only optional.
# morpho = MorphoDataset("czech_pdt_lemmas", add_bow_eow=True)
# analyses = MorphoAnalyzer("czech_pdt_analyses")


# def tagging_dataset(example, vectors):
#     return (example["forms"], example["lemmas"], vectors)
#
#
# def create_dataset(name, vectors, shuffling=True):
#     dataset = getattr(morpho, name).dataset
#     vecset = tf.data.Dataset.from_tensor_slices(vectors)
#     dataset = tf.data.Dataset.zip((dataset, vecset))
#     dataset = dataset.map(tagging_dataset)
#     if shuffling:
#         dataset = dataset.shuffle(args.batch_size * 10, seed=args.seed) if name == "train" else dataset
#     dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
#     return dataset


# # train, dev, test = create_dataset("train"), create_dataset("dev"), create_dataset("test")
# train = create_dataset('train', train_vecs, shuffling=True)
# dev = create_dataset('dev', dev_vecs, shuffling=True)
# test = create_dataset('test', test_vecs, shuffling=True)

#print(len(train))
scheduler = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001, decay_steps=180000 * args.epochs
)

checkpoint_path = os.path.join(args.logdir, "checkpoint.weights")
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 mode="max",
                                                 monitor="val_accuracy",
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 verbose=1)

# TODO: Create the model and train it
model = Model(args, scheduler, morpho.train)

logs = model.fit(train, epochs=args.epochs, validation_data=dev, verbose=2,
                 callbacks=[model.tb_callback, cp_callback])

def make_sparse(curr_word):
    vector = tf.constant(curr_word)
    split = tf.strings.bytes_split(vector, 'UTF-8')
    return tf.sparse.from_dense(split[None, :])

os.makedirs(args.logdir, exist_ok=True)

test_path_simple = os.path.join(args.logdir, "lemmatizer_test_simple.txt")
dev_path_simple = os.path.join(args.logdir, "lemmatizer_dev_simple.txt")

test_path_analysis = os.path.join(args.logdir, "lemmatizer_test_analysis.txt")
dev_path_analysis = os.path.join(args.logdir, "lemmatizer_dev_analysis.txt")


def predict_simple(path, dataset):
    with open(path, "w", encoding="utf-8") as predictions_file:
        predictions = model.predict(dataset)
        for sentence in predictions:
            for word in sentence:
                print(word.numpy().decode("utf-8"), file=predictions_file)
            print(file=predictions_file)


predict_simple(path=test_path_simple, dataset=test)
predict_simple(path=dev_path_simple, dataset=dev)

gold = MorphoDataset('czech_pdt_lemmas').dev.lemmas

with open(dev_path_simple, 'rt', encoding='utf-8') as f:
    print('dev simple:', MorphoDataset.evaluate_file(gold, f))

with open(dev_path_analysis, 'rt', encoding='utf-8') as f:
    print('dev analysis:', MorphoDataset.evaluate_file(gold, f))
