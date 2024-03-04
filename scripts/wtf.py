from bpemb import BPEmb
import numpy as np


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


print("Loading model for static embeddings...")
multibpe_model = BPEmb(lang='multi',
                       vs=100000,
                       dim=300)

leave = False

while leave is False:
    user_input = input("Word: ").strip().split(" ")

    if len(user_input) == 1:
        if user_input == "Q":
            leave = True
            continue
        else:
            input_word = user_input[0]
            input_language = "UNK"

    elif len(user_input) == 2:
        input_word, input_language = user_input

    else:
        print("Invalid input! Please enter a maximum of 2 words, the second of which should be a language.")
        continue

    print(bpe_encode(input_word, multibpe_model))