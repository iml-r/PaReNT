#!/usr/bin/env python3

import re
import sys



LATIN_RE = re.compile(r"[A-Za-zěščřýžýáíéúůďťňóĚŠČŘÝŽÝÁÍÉÚŮĎŤŇÓöüäëèñĺçÖÜÄľĽËÈÑĹÇ]")


CYRIL_RE = re.compile(r"[АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя]")


SOFT = ["Ь", "ь"]


TABLE = {
    # CAPITAL LETTERS
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "JE",
    "Ж": "Ž", "З": "Z", "И": "I", "Й": "J",
    "К": "K", "Л": "L", "М": "M", "Н": "N", "О": "O", "П": "P", "Р": "R",
    "С": "S", "Т": "T", "У": "U", "Ф": "F", "Х": "CH", "Ц": "C", "Ч": "Č",
    "Ш": "Š", "Щ": "ŠČ", "Ы": "Y", "Э": "E", "Ю": "JU", "Я": "JA", "H": "H",
    "Ё": "JO",

    # LOWERCASE LETTERS
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "je",
    "ж": "ž", "з": "z", "и": "i", "й": "j",
    "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "ch", "ц": "c", "ч": "č",
    "ш": "š", "щ": "šč", "ы": "y", "э": "e", "ю": "ju", "я": "ja", "h": "h",
    "ё": "jo"
}


SOFTENING = {
    "D": "Ď", "T": "Ť", "N": "Ň",
    "d": "ď", "t": "ť", "n": "ň",
}

NON_JOINING = ["цг", "йа", "ый", "йе", "йу", "шч"]


def cyril_to_latin(text):
    # HANDLE DOUBLE LETTERS THAT SHOULD BE KEPT
    for pair in NON_JOINING:
        # 1. Escape them if there are sepeterate by dash already
        text = text.replace(pair[0] + "-" + pair[1], pair[0] + "---" + pair[1])
        text = text.replace(pair[0].upper() + "-" + pair[1], pair[0].upper() + "---" + pair[1])
        text = text.replace(pair[0].upper() + "-" + pair[1].upper(), pair[0].upper() + "---" + pair[1].upper())

        # 2. Add dash not to merge them
        text = text.replace(pair[0] + pair[1], pair[0] + "-" + pair[1])
        text = text.replace(pair[0].upper() + pair[1], pair[0].upper() + "-" + pair[1])
        text = text.replace(pair[0].upper() + pair[1].upper(), pair[0].upper() + "-" + pair[1].upper())

    output = []

    if LATIN_RE.search(text) and CYRIL_RE.search(text) is None:
        return f"{text}"

    for i, char in enumerate(text):
        if LATIN_RE.search(char):
            output.append(f"{char}")
            continue

        trans = TABLE.get(char, char)
        if len(trans) > 1 and i < len(text) - 1 and text[i + 1].islower():
            trans = trans[0] + trans[1:].lower()

        if trans.lower() == "je" and output and output[-1] in SOFTENING:
            if char == "Е":
                trans = "Ě"
            if char == "е":
                trans = "ě"

        if trans.lower() == "ja" and output and output[-1] in SOFTENING:
            output[-1] = SOFTENING.get(output[-1], output[-1])
            if char == "Я":
                trans = "A"
            if char == "я":
                trans = "a"

        if trans.lower() == "ju" and output and output[-1] in SOFTENING:
            output[-1] = SOFTENING.get(output[-1], output[-1])
            if char == "Ю":
                trans = "U"
            if char == "ю":
                trans = "u"

        # if trans.lower() == "ji" and output and output[-1] in SOFTENING:
        #     output[-1] = SOFTENING.get(output[-1], output[-1])
        #     if char == "Ї":
        #         trans = "I"
        #     if char == "ї":
        #         trans = "i"

        if char in SOFT and output:
            if output[-1] in SOFTENING:
                output[-1] = SOFTENING.get(output[-1], output[-1])
            else:
                output.append(char)
            continue

        output.append(trans)
    return "".join(output)


def main():
    for line in sys.stdin:
        words = []
        for word in line.strip().split():
            words.append(cyril_to_latin(word))
        print(" ".join(words))


if __name__ == "__main__":
    main()
