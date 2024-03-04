#!/usr/bin/env python3

import sys

UNIGRAM_TABLE = {
    "A": "А", "B": "Б", "V": "В", "G": "Г", "D": "Д", "E": "Э",
    "Ž": "Ж", "Z": "З", "I": "И", "J": "Й", "K": "К", "L": "Л",
    "M": "М", "N": "Н", "O": "О", "P": "П", "R": "Р", "S": "С", "T": "Т",
    "U": "У", "F": "Ф", "C": "Ц", "Č": "Ч", "Š": "Ш", "Y": "Ы", "Ě": "Е", "Ď": "ДЬ",
    "Ť": "ТЬ", "Ň": "НЬ", "H": "H",

    "a": "а", "b": "б", "v": "в", "g": "г", "d": "д", "e": "э",
    "ž": "ж", "z": "з", "y": "ы", "i": "и", "j": "й", "k": "к", "l": "л",
    "m": "м", "n": "н", "o": "о", "p": "п", "r": "р", "s": "с", "t": "т",
    "u": "у", "f": "ф", "c": "ц", "č": "ч", "š": "ш", "ě": "е", "ď": "дь",
    "ť": "ть", "ň": "нь", "ˇ": "ь", "h": "h"
}

BIGRAM_TABLE = {
    "ŠČ": "Щ", "JU": "Ю", "JA": "Я", "CH": "Х", "JE": "Е",
    "ĎA": "ДЯ", "ŤA": "ТЯ", "ŇA": "НЯ", "ĎI": "ДИ", "ŤI": "ТИ", "ŇI": "НИ",
    "ĎU": "ДЮ", "ŤU": "ТЮ", "ŇU": "НЮ",
    "JO": "Ё",

    "Šč": "Щ", "Ju": "Ю", "Ja": "Я", "Ch": "Х", "Je": "Є",
    "Ďa": "Дя", "Ťa": "Тя", "Ňa": "Ня", "Ďi": "Ди", "Ťi": "Ти", "Ňi": "Ни",
    "Ďu": "Дю", "Ťu": "Тю", "Ňu": "Ню",
    "Jo": "Ё",

    "šč": "щ", "ju": "ю", "ja": "я", "ch": "х", "je": "е",
    "ďa": "дя", "ťa": "тя", "ňa": "ня", "ďi": "ди", "ťi": "ти", "ňi": "ни",
    "ďu": "дю", "ťu": "тю", "ňu": "ню",
    "jo": "ё",
}

NON_JOINING = ["цг", "йа", "ый", "йе", "йу", "шч"]


def latin_to_cyril(text):
    is_upper = (
        len([c for c in text if c.isupper()]) >
        len([c for c in text if c.islower()]))


    output = []
    i = 0
    while i < len(text):
        if text[i:].startswith("<latin_char>"):
            output.append(text[i + 12])
            i += 13
            continue

        if i < len(text) - 1:
            bigram = text[i:i + 2]

            if bigram in BIGRAM_TABLE:
                trans = BIGRAM_TABLE[bigram]
                if is_upper and len(trans) > 1:
                    trans = trans.upper()
                i += 2
                if trans.islower() and output and output[-1].isupper() and len(output[-1]) > 1:
                    output[-1] = output[-1][0] + output[-1][1:].lower()
                output.append(trans)
                continue

        unigram = text[i]
        trans = UNIGRAM_TABLE.get(unigram, unigram)
        if is_upper and (len(trans) > 1 or trans == 'ь'):
            trans = trans.upper()
        if trans.islower() and output and output[-1].isupper() and len(output[-1]) > 1:
            output[-1] = output[-1][0] + output[-1][1:].lower()
        output.append(trans)
        i += 1

    out_str = "".join(output)

    for pair in NON_JOINING:
        # 1. Escape them if there are sepeterate by dash already
        out_str = out_str.replace(pair[0] + "-" + pair[1], pair[0] + pair[1])
        out_str = out_str.replace(
            pair[0].upper() + "-" + pair[1], pair[0].upper() + pair[1])
        out_str = out_str.replace(
            pair[0].upper() + "-" + pair[1].upper(),
            pair[0].upper() + pair[1].upper())

        # 2. Add dash not to merge them
        out_str = out_str.replace(
            pair[0] + "---" + pair[1], pair[0] + "-" + pair[1])
        out_str = out_str.replace(
            pair[0].upper() + "---" + pair[1], pair[0].upper() + "-" + pair[1])
        out_str = out_str.replace(
            pair[0].upper() + "---" + pair[1].upper(),
            pair[0].upper() + "-" + pair[1].upper())

    return out_str


def main():
    for line in sys.stdin:
        words = []
        for word in line.strip().split():
            words.append(latin_to_cyril(word))
        print(" ".join(words))


if __name__ == "__main__":
    main()
