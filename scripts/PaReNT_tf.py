#!/usr/bin/python3

import os
import datetime
import argparse
# import re
# import random
# import funcy

import tensorflow as tf
import tensorflow_addons as tfa
# import tensorflow_addons as tfa
# import tensorflow_models as tfm
# from ipython_genutils.py3compat import input
# from numpy import bytes_
# from official.nlp.modeling.losses.weighted_sparse_categorical_crossentropy import loss
from transformers import AutoTokenizer, TFT5ForConditionalGeneration, WarmUp
import numpy as np
import pandas as pd
from tqdm import tqdm

##Create logdir name
# logdir = os.path.join("logs", "{}-{}-{}".format(
#     os.path.basename(globals().get("__file__", "notebook")),
#     datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
#     ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
# ))
# #

##Data preprocessing
# @tf.function
# def split(x):
#     return tf.reshape(tf.cast(tf.io.decode_raw(x, tf.uint8), tf.int32), [1, -1])
#
# train = tf.data.experimental.make_csv_dataset(file_pattern="data/train/*",
#                                               batch_size=1,
#                                               header=True,
#                                               field_delim=".",
#                                               select_columns=["lexeme", "parents"])
# train = train.map(lambda a: (a['lexeme'], split(a['parents'])))
# #train = train.apply(tf.data.experimental.dense_to_ragged_batch(512))
#
# dev = tf.data.experimental.make_csv_dataset(file_pattern="data/validate/*",
#                                               batch_size=1,
#                                               header=True,
#                                               field_delim=".",
#                                               select_columns=["lexeme", "parents"])
# dev = dev.map(lambda a: (a['lexeme'], split(a['parents'])))

##Argparse
parser = argparse.ArgumentParser(description='Model hyperparameters.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size')
parser.add_argument('--epochs', type=int, default=20,
                    help='Epochs')
parser.add_argument('--lr', type=float, default=3e-4,
                    help='Learning rate')
parser.add_argument('--decay', type=int, default=0,
                    help='Learning rate')
parser.add_argument('--decay_steps', type=int, default=10000,
                    help='Decay rate')
parser.add_argument('--weight_decay', type=float, default=0.0000000001,
                    help='Weight decay')
parser.add_argument('--loss', type=str, default="normal",
                    help='Weight decay')

args = parser.parse_args()

print("Epochs: ", args.epochs)
##
tf.argmax([1])
##Utils
def longest(lst):
    max_length = 0

    for word in lst:
        length = len(word.encode('utf-8'))
        if length > max_length:
            max_length = length

    return max_length

def prepend_int_matrix(tensor: tf.Tensor, bos_token_id: tf.int32) -> tf.Tensor:
    assert tf.rank(tensor) == 2
    if tensor.dtype != tf.int32:
        print("Warning: Recasting tensor input of the integer matrix prepending function into tf.int32!")
        tensor = tf.cast(tensor, tf.int32)

    token_tensor = tf.fill([tf.shape(tensor)[0], 1], bos_token_id)
    return(tf.concat([token_tensor, tensor], axis=1))

class TFCausalLanguageModelingLoss:
    """
    Loss function suitable for causal language modeling (CLM), that is, the task of guessing the next token.
    <Tip>
    Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.
    </Tip>
    """

    def hf_compute_loss(self, labels, logits):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        # if self.config.tf_legacy_loss:
        #     # make sure only labels that are not equal to -100 affect the loss
        #     active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
        #     reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
        #     labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
        #     return loss_fn(labels, reduced_logits)

        # Clip negative labels to zero here to avoid NaNs and errors - those positions will get masked later anyway
        unmasked_loss = loss_fn(tf.nn.relu(labels), logits)
        # make sure only labels that are not equal to -100 affect the loss
        loss_mask = tf.cast(labels != -100, dtype=unmasked_loss.dtype)
        masked_loss = unmasked_loss * loss_mask
        reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask)
        return tf.reshape(reduced_masked_loss, (1,))

causal_loss = TFCausalLanguageModelingLoss()
##



#Set distribution strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
##



##Pretrained models and tokenizer
print("Loading model...")

with strategy.scope():
    byt5 = TFT5ForConditionalGeneration.from_pretrained('google/byt5-small')
    byt5.trainable = True
    metrics = [tf.metrics.SparseCategoricalAccuracy()]

    ##Model options
    if args.decay == 1:
        print("Decay without warmup")
        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(args.lr,
                                                                  args.decay_steps)
    elif args.decay == 2:
        print("Decay with warmup")
        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(args.lr,
                                                                  args.decay_steps)
        lr_decayed_fn = WarmUp(initial_learning_rate=args.lr,
                               decay_schedule_fn=lr_decayed_fn,
                               warmup_steps=5e4)
    else:
        print("Constant lr")
        lr_decayed_fn = args.lr

    if args.loss == "normal":
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        loss = causal_loss.hf_compute_loss

    optimizer = tfa.optimizers.AdamW(learning_rate=lr_decayed_fn,
                                     weight_decay=args.weight_decay)

    byt5.compile(optimizer=optimizer,
                 metrics=metrics,
                 loss=loss)

    tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
    tokenizer.bos_token = "</b>"
    byt5.config.bos_token_id = tokenizer.bos_token_id
print(byt5.config.bos_token_id)
##



##Training
print("Loading training data...")
pdlist = []
file_path = "../data/train/"
for i in tqdm(os.listdir(file_path)):
    pdlist.append(pd.read_csv(file_path+i, sep="."))
train = pd.concat(pdlist)
del pdlist

train = train.sample(frac=1)
#FIXME:
train = train.dropna()

# train_x = list(funcy.partition(64, list(train['lexeme'])))
# train_Y = list(funcy.partition(64, list(train['parents'])))

##https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/t5#training
train_x = list(train['lexeme'])
train_x = [i for i in train_x]
max_source_length = longest(train_x)

train_Y = list(train['parents'])
train_Y = [i for i in train_Y]
max_target_length = longest(train_Y)

maxlen = np.max([max_source_length, max_target_length])

print("Encoding...")
encoding_x = tokenizer(
    train_x,
    padding='max_length',
    max_length=maxlen,
    truncation=True,
    return_tensors="tf",
)

input_ids_x, attention_mask_x = encoding_x.input_ids, encoding_x.attention_mask
input_ids_x, attention_mask_x = prepend_int_matrix(input_ids_x, 2), prepend_int_matrix(attention_mask_x, 2)

encoding_Y = tokenizer(
    train_Y,
    padding='max_length',
    max_length=maxlen,
    truncation=True,
    return_tensors="tf"
)

labels = encoding_Y.input_ids
labels = prepend_int_matrix(labels, 2)

decoder_ids = encoding_Y.input_ids
decoder_ids = prepend_int_matrix(decoder_ids, 2)

input_ids_x = tf.data.Dataset.from_tensor_slices(input_ids_x)
attention_mask_x = tf.data.Dataset.from_tensor_slices(attention_mask_x)
decoder_ids = tf.data.Dataset.from_tensor_slices(decoder_ids)
labels = tf.data.Dataset.from_tensor_slices(labels)
#labels = tf.data.Dataset.from_tensor_slices((tf.cast(labels == tokenizer.pad_token_id, tf.int32))*(-100) + labels)
#print(type(labels))

z = tf.data.Dataset.zip((input_ids_x, attention_mask_x, decoder_ids))
inputs = tf.data.Dataset.zip((z, labels)).batch(args.batch_size)
#print([*inputs.take(10).as_numpy_iterator()])
inputs_dist = strategy.experimental_distribute_dataset(inputs)

##Validation data
print("Loading validation data...")
file_path = "../data/validate/"

pdlist = []
for i in tqdm(os.listdir(file_path)):
    pdlist.append(pd.read_csv(file_path+i, sep="."))
val = pd.concat(pdlist)
val = val.dropna()
del pdlist

val_x = list(val['lexeme'])
val_x = [tokenizer.bos_token+i for i in val_x]
max_source_length = longest(val_x)

val_Y = list(val['parents'])
val_Y = [tokenizer.bos_token+i for i in val_Y]
max_target_length = longest(val_Y)
maxlen = np.max([max_source_length, max_target_length])

print("Encoding...")
encoding_val_x = tokenizer(
    val_x,
    padding='max_length',
    max_length=maxlen,
    truncation=True,
    return_tensors="tf"
)

input_ids_val_x, attention_mask_val_x = encoding_val_x.input_ids, encoding_val_x.attention_mask
del val_x

encoding_val_Y = tokenizer(
    val_Y,
    padding='max_length',
    max_length=maxlen,
    truncation=True,
    return_tensors="tf"
)

decoder_val_ids = encoding_val_Y.input_ids
labels_val = encoding_val_Y.input_ids
del val_Y



inputs_ids_val_x = tf.data.Dataset.from_tensor_slices(input_ids_val_x)
attention_mask_val_x = tf.data.Dataset.from_tensor_slices(attention_mask_val_x)
decoder_ids_val_x = tf.data.Dataset.from_tensor_slices(decoder_val_ids)

labels_val = tf.data.Dataset.from_tensor_slices(labels_val)
#labels_val = tf.data.Dataset.from_tensor_slices((tf.cast(labels_val == tokenizer.pad_token_id, tf.int32))*(-100) + labels_val)
z = tf.data.Dataset.zip((inputs_ids_val_x, attention_mask_val_x, decoder_ids_val_x))

val_inputs = tf.data.Dataset.zip((z, labels_val)).batch(args.batch_size)
val_inputs_dist = strategy.experimental_distribute_dataset(val_inputs)

##Actual training process

# train_x = [tokenizer.prepare_seq2seq_batch(i) for i in train_x]
# train_Y = [tokenizer.prepare_seq2seq_batch(i) for i in train_Y]
epochs = args.epochs
weights = {i:1 for i in range(tokenizer.vocab_size)}
weights[1] = 3.
weights[0] = .1

date = str(datetime.date.today().isocalendar()[:]).replace("(", "").replace(")", "").replace(", ", "-")
sample_data = ["černobřichý",
               "development",
               "seven-year-old",
               "Liebe",
               "Liebesbrief",
               "ящик"]

input_sample = tokenizer(sample_data,
                  padding='max_length',
                  max_length=maxlen,
                  truncation=True,
                  return_tensors="tf", )

sample_data = input_sample.input_ids

sample_test_data = ["andragogika",
               "cubresuelos",
               "bavardocher",
               "zyrillisch",
               "Zypressenholz",
               "ёрзать"]

input_test_sample = tokenizer(sample_test_data,
                  padding='max_length',
                  max_length=maxlen,
                  truncation=True,
                  return_tensors="tf", )


for i in tqdm(range(epochs)):
    print(f"\n Epoch: {i+1}")
    # print(decoder_ids)
    #print(tokenizer.batch_decode(byt5.generate(input_sample.input_ids)))
    # print("Train sample:", tokenizer.decode([(inputs.take(1).as_numpy_iterator()][0][0]))
    # print("Val sample:", tokenizer.decode([*val_inputs.take(1).as_numpy_iterator()][0]))


    byt5.fit(inputs,
             #steps_per_epoch=inputs_dist.cardinality//args.batch_size,
             validation_data=val_inputs,
             #validation_steps=val_inputs_dist.cardinality//args.batch_size,
             epochs=1,
             class_weight=weights)
    modelname = "../tf_models/" + date + f"/byt5_{date}_e{i+1}_b{args.batch_size}_wd{args.weight_decay}_lr{args.lr}_dec{args.decay}"
    print(f"Saving model name: {modelname}...")
    byt5.save_pretrained(modelname)

    print("input:",["černobřichý",
     "development",
     "seven-year-old",
     "Liebe",
     "Liebesbrief",
     "ящик"])
    print("Generation:", tokenizer.batch_decode(byt5.generate(sample_data,
                                                              no_repeat_ngram_size=2,
                                                              early_stopping=True,
                                                              repetition_penalty=1.5), skip_special_tokens=True))

    sample_input = prepend_int_matrix(sample_data, 2)
    print("Generation with prepended BOS token:", tokenizer.batch_decode(byt5.generate(sample_input,
                                                              no_repeat_ngram_size=2,
                                                              early_stopping=True,
                                                              repetition_penalty=1.5), skip_special_tokens=True))

    print(["andragogika",
               "cubresuelos",
               "bavardocher",
               "zyrillisch",
               "Zypressenholz",
               "ёрзать"]
)
    print("Generation with prepended BOS token, test data:", tokenizer.batch_decode(byt5.generate(input_test_sample.input_ids,
                                                                                                   no_repeat_ngram_size=2,
                                                                                                   early_stopping=True,
                                                                                                   repetition_penalty=1.5),
                                                                                                   skip_special_tokens=True))

print("Done!")
#
# ##Model architecture
# input = tf.keras.layers.Input([1],
#                               dtype=tf.string)
#
# single_characters = tf.keras.layers.Lambda(
#                                     function=split,
#                                     output_shape=[None])(input)
# single_characters = tf.keras.layers.Reshape(target_shape=[-1])(single_characters)
# byt5_layer0 = byt5.layers[0](single_characters)
# byt5_layer1 = byt5.layers[1]({'inputs_embeds': byt5_layer0}).last_hidden_state
# byt5_layer2 = byt5.layers[2]({'inputs_embeds': byt5_layer1})
#
# fully_connected = tf.keras.layers.Dense(units=2048, activation=tf.nn.swish)
# out = tf.keras.layers.TimeDistributed(fully_connected)(byt5_layer2[0])
#
# outputs = tf.keras.layers.Dense(units=(2**8)+1,
#                                      activation=None)(out)
#
# ##
#
# ##Model
# # model = tf.keras.Model(inputs=input,
# #                        outputs=outputs)
#
# class PaReNT(tf.keras.Model):
#     def __init__(self) -> None:
#         super().__init__()
#
#         self.inputs = tf.keras.layers.Input([1], dtype=tf.string)
#
#         self.single_characters = tf.keras.layers.Lambda(
#             function=split,
#             output_shape=[None])(self.inputs)
#         self.single_characters = tf.keras.layers.Reshape(target_shape=[-1])(self.single_characters)
#         self.byt5_layer0 = byt5.layers[0](self.single_characters)
#         self.byt5_layer1 = byt5.layers[1]({'inputs_embeds': self.byt5_layer0}).last_hidden_state
#         self.byt5_layer2 = byt5.layers[2]({'inputs_embeds': self.byt5_layer1})
#
#         self.fully_connected = tf.keras.layers.Dense(units=2048, activation=tf.nn.swish)
#         self.out = tf.keras.layers.TimeDistributed(self.fully_connected)(self.byt5_layer2[0])
#
#         self.outputs = tf.keras.layers.Dense(units=(2 ** 8) + 1,
#                                         activation=None)(self.out)
#
#         # Compile the model
#         self.compile(
#             optimizer=tf.optimizers.Adam(),
#             loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
#             metrics=[tf.metrics.Accuracy(name="accuracy")],
#         )
#
#         #self.tb_callback = tf.keras.callbacks.TensorBoard(logdir)
#
#     # If `targets` is given, we are in the teacher forcing mode.
#     # Otherwise, we run in autoregressive mode.
#     def call(self, inputs, targets=None):
#         #inputs = self.inputs(inputs)
#         output = self.outputs(inputs)
#
#         return output
#
#     def train_step(self, data):
#         # x, y = data
#         #
#         # # Convert `y` by splitting characters, mapping characters to ids using
#         # # `self.target_mapping` and finally appending `MorphoDataset.Factor.EOW`
#         # # to every sequence.
#         # y_targets = y
#         # y_targets = tf.concat(
#         #     [y_targets, tf.fill([y_targets.bounding_shape(0), 1],
#         #                         tf.constant(256, tf.int64))], axis=-1)
#         # y_targets = y.with_values(y_targets)
#         #
#         # with tf.GradientTape() as tape:
#         #     y_pred = self(x, targets=y_targets, training=True)
#         #     loss = self.compute_loss(x, y_targets.flat_values, y_pred.flat_values)
#         # self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
#         # return {"loss": metric.result() for metric in self.metrics if metric.name == "loss"}
#
#         return {}
#
#     def predict_step(self, data):
#         if isinstance(data, tuple):
#             data = data[0]
#         y_pred = self(data, training=False)
#         #y_pred = self.target_mapping_inverse(y_pred)
#         #y_pred = tf.strings.reduce_join(y_pred, axis=-1)
#         return y_pred
#
#     def test_step(self, data):
#         # x, y = data
#         # y_pred = self.predict_step(data)
#         # self.compiled_metrics.update_state(
#         #     tf.ones_like(y.values, dtype=tf.int32), tf.cast(y_pred.values == y.values, tf.int32))
#         # return {m.name: m.result() for m in self.metrics if m.name != "loss"}
#
#         pass
#
#
# ##
#
#
# ##Optimizer
# optimizer = tf.keras.optimizers.Adam()
# ##
#
# ##Loss function
# def VariableLength_SparseCategoricalCrossentropy(y_true, y_pred):
#     # tf.print(y_true.shape, y_pred.shape)
#     # print(y_true.shape, y_pred.shape)
#
#     true_shape = tf.shape(y_true)
#     pred_shape = tf.shape(y_pred)
#     scce = tf.losses.SparseCategoricalCrossentropy(from_logits=True,
#                                             reduction=tf.keras.losses.Reduction.SUM)
#
#     if true_shape[1] == pred_shape[1]:
#         return scce(y_true, y_pred)
#
#     elif true_shape[1] > pred_shape[1]:
#         shape = [pred_shape[0], true_shape[1] - pred_shape[1], pred_shape[2]]
#
#         # shape = list(tf.shape(y_pred).numpy())
#         # shape[1] = (pred_shape[1] - true_shape[1]).numpy()
#         new_y_pred = tf.concat([y_pred, tf.zeros(shape)], axis=1)
#         return scce(y_true, new_y_pred)
#
#     elif true_shape[1] < pred_shape[1]:
#         #shape = [true_shape[0], pred_shape[1] - true_shape[1]]
#         shape = [true_shape[0], pred_shape[1] - true_shape[1], true_shape[2]]
#         new_y_true = tf.concat([y_true, tf.fill(shape, 256)], axis=1)
#         return scce(new_y_true, y_pred)
#
#     else:
#         return tf.constant(0.)
#
# # def VariableLength_SparseCategoricalAccuracy(y_true, y_pred):
# #     # tf.print(y_true.shape, y_pred.shape)
# #     # print(y_true.shape, y_pred.shape)
# #
# #     true_shape = tf.shape(y_true)
# #     pred_shape = tf.shape(y_pred)
# #     acc = tf.metrics.SparseCategoricalAccuracy()
# #
# #     if true_shape[1] == pred_shape[1]:
# #         return acc(y_true, y_pred)
# #
# #     elif true_shape[1] > pred_shape[1]:
# #         shape = [pred_shape[0], true_shape[1] - pred_shape[1], pred_shape[2]]
# #
# #         # shape = list(tf.shape(y_pred).numpy())
# #         # shape[1] = (pred_shape[1] - true_shape[1]).numpy()
# #         new_y_pred = tf.concat([y_pred, tf.zeros(shape)], axis=1)
# #         return acc(y_true, new_y_pred)
# #
# #     elif true_shape[1] < pred_shape[1]:
# #         shape = [true_shape[0], pred_shape[1] - true_shape[1]]
# #         new_y_true = tf.concat([y_true, tf.fill(shape, 256.)], axis=1)
# #         return acc(new_y_true, y_pred)
# #
# #     else:
# #         return tf.constant(0.)
#
# ##
#
# ##Weights
# weights = tf.fill([2**8], 1.0)
# weights = tf.concat([weights, tf.constant([0.0])], axis=0)
# ##
#
# model.compile(optimizer=optimizer,
#               loss=tf.losses.SparseCategoricalCrossentropy(),
#               )
#
# model.fit(train,
#           validation_data=dev,
#           epochs=2)
#

#yo

