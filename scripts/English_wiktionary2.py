import sys
import logging
import argparse
import regex as re
from bz2 import BZ2File
import xml.etree.ElementTree as ET


def load_relevant_entries(path, language_content):
    entries = list()
    with BZ2File(path) as xml_file:
        parser = ET.iterparse(xml_file)
        open_extraction = False

        for event, element in parser:
            # get the head lemma
            if element.tag[43:] == 'title':
                title = element.text

            # check xml element, open saving only if it is lemma
            elif element.tag[43:] == 'ns' and element.text == '0':
                open_extraction = True

            # extract information about lemma
            elif element.tag[43:] == 'text' and open_extraction is True:
                content = re.search(language_content[0], element.text)
                if content:
                    # find out in how many langugaes the word is used
                    # and extract the relevant one only
                    number_of_langugaes = re.findall(
                        language_content[1], content.group()
                    )
                    if len(number_of_langugaes) > 1:
                        content = re.search(
                            language_content[2], content.group()
                        )
                        if not content:
                            continue

                    # store lemma and its content
                    entries.append((title, content.group()))
                open_extraction = False

            element.clear()
    return entries
def extract_information(data, patterns):
    entries = list()  # [['lemma', 'parent'], ...]
    for lemma, content in data:
        for pattern in patterns:
            # extract information about the desired category
            desired_content = re.search(pattern, content)

            # extract derivational parent of the main lemma
            if desired_content:
                entries.append((lemma, desired_content.group(1)))
    return entries

xml = load_relevant_entries("English/enwiktionary-20210601-pages-articles.xml.bz2", language_content=(
            r'==English==\n(.*\n)*.*',
            r'((?<!=)==[A-Z])',
            r'==English==\n(.*\n)*?(----)'
        ))

patterns = (r'{{compound\|en\|[a-z|]+}}',)
entries = extract_information(xml, patterns)

import re
import pandas as pd

pattern = re.compile(r'{{compound\|en\|[a-z|]+}}')
pagelist2 = [i for i in xml if pattern.search(i[1].replace("\n", " "))]

print("data filtered")

df = pd.DataFrame()
kompozita = []
rodice = []

for i in pagelist2:
    kompozita.append(i[0])
    match = pattern.search(i[1])
    rodice.append(match[0].replace("{{compound|en|", "").replace("}}", ""))

rodice = [i.replace("|", " ") for i in rodice]

df['kompozitum'] = kompozita
df['rodiče'] = rodice

print("data ready - writing...")

df.to_csv("English/wiktionary_mined.tsv", sep="\t")