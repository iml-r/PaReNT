#!/usr/bin/env python3
# coding: utf-8

import sys
import logging
import argparse
import regex as re
from bz2 import BZ2File
import xml.etree.ElementTree as ET


# initialise arguments
parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, action='store')
parser.add_argument('--language_mutation_of_wikti', type=str, action='store')
parser.add_argument('--language_to_extract', type=str, action='store')
parser.add_argument('--input_data', type=str, action='store')
parser.add_argument('--output_data', type=str, action='store')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')


# load relevant data from wiktionary
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


# extract relevant information related to desired semantic category
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


# cleaning function for strings of derivational parent
def cs_cs_clean_parent_strings(data):  # Czech from Czech Wiktionary
    entries = list()
    for item in data:
        # initialise
        entry = list(item[:1])

        # clean
        parent = re.findall(r'\[\[.*?\]\]', item[1])
        if len(parent) > 0:
            parent = parent[0].replace('[', '').replace(']', '')
        else:
            parent = ''

        # save
        entry.append(parent)
        entries.append(tuple(entry))
    return entries


def en_cs_clean_parent_strings(data):  # Czech from English Wiktionary
    entries = list()
    for item in data:
        # initialise
        entry = list(item[:1])

        # clean
        parent = item[1]

        # save
        entry.append(parent)
        entries.append(tuple(entry))
    return entries


def fr_fr_clean_parent_strings(data):  # French from French Wiktionary
    entries = list()
    for item in data:
        # initialise
        entry = list(item[:1])

        # clean TODO
        parent = item[1]
        if ' ' in parent:
            parent = ''

        if ' ' in entry[0]:
            continue

        # save
        entry.append(parent)
        entries.append(tuple(entry))
    return entries


def en_fr_clean_parent_strings(data):  # French from English Wiktionary
    entries = list()
    for item in data:
        # initialise
        entry = list(item[:1])

        # clean
        parent = re.sub(r'\|\|.*', '', item[1])
        try:
            cl_parent = re.search(r'^(.*?)\|', parent[::-1]).group(1)[::-1]
            parent = cl_parent
        except AttributeError:
            continue
        if parent in ('fr', 'fro', 'la', 'inh', 'der', 'etyl', 'xno', 'pro',
                      'non', 'vík', 'm', 'dum', 'it', 'pt', 'ru', 'br') or \
           '=' in parent:
            parent = ''

        if ' ' in entry[0]:
            continue

        # save
        entry.append(parent)
        entries.append(tuple(entry))
    return entries

def ru_ru_clean_parent_strings(data):  # Russian from Russian Wiktionary
    entries = list()
    for item in data:
        # initialise
        entry = list(item[:1])

        # clean
        parent = item[1]

        # save
        entry.append(parent)
        entries.append(tuple(entry))
    return entries

def en_ru_clean_parent_strings(data):  # Russian from English Wiktionary
    entries = list()
    for item in data:
        # initialise
        entry = list(item[:1])

        # clean
        parent = item[1]

        # save
        entry.append(parent)
        entries.append(tuple(entry))
    return entries

# main script
def main(args):
    # set right settings for the extraction
    if (  # Czech from Czech Wikti
        args.language_mutation_of_wikti == 'cs' and
        args.language_to_extract == 'cs'
       ):
        language_content = (
            r'== čeština ==\n(.*\n)*',
            r'((?<!=)== )',
            r'== čeština ==\n(.*\n)*?(== )'
        )
        if args.category == 'diminutives':
            patterns = (
                r'zdrob\.(.*)',
            )
        else:
            logging.error('Unknown combination of semantic category settings.')
            sys.exit()

    elif (  # Czech from English Wikti
          args.language_mutation_of_wikti == 'en' and
          args.language_to_extract == 'cs'
         ):
        language_content = (
            r'==Czech==\n(.*\n)*.*',
            r'((?<!=)==[A-Z])',
            r'==Czech==\n(.*\n)*?(----)'
        )
        if args.category == 'diminutives':
            patterns = (
                r'diminutive of\|cs\|(.*?)[\}\|]',
            )
        else:
            logging.error('Unknown combination of semantic category settings.')
            sys.exit()

    elif (  # TODO: French from French Wikti
          args.language_mutation_of_wikti == 'fr' and
          args.language_to_extract == 'fr'
         ):
        language_content = (
            r'== \{\{langue\|fr\}\} ==\n(.*\n)*.*',
            r'== \{\{langue\|',
            r'== \{\{langue\|fr\}\} ==\n(.*\n)*?(== \{\{langue\|)'
            r'==French==\n(.*\n)*?(----)'
        )
        if args.category == 'diminutives':
            patterns = (
                r'Diminutif.*?\[\[(\p{L}.*?)[\|\]#\}]',
                r'diminutif de .*?\[\[(\p{L}.*?)[\]#\|\}]',
                r'\[\[formé\|Formé\]\] de petits* \[\[(.*?)[\]#\}\|]',
                r'Toute petite* \[\[(.*?)\]\].',
                r'# \[\[très\|Très\]\] jeunes* \[\[(.*?)[\]\|]',
                r'# [Pp]etite* \[\[(.*?)\]\]\.',
                # r'# .*[jJ]eunes* \[*\[*(.*?)[\]\|\}]',
                # r'==== \{\{S\|traductions\}\} ====\n\{\{trad-début\|Petite* (.*?)[\]\|\.#\}]'
            )
        else:
            logging.error('Unknown combination of semantic category settings.')
            sys.exit()

    elif (  # French from English Wikti
          args.language_mutation_of_wikti == 'en' and
          args.language_to_extract == 'fr'
         ):
        language_content = (
            r'==French==\n(.*\n)*.*',
            r'((?<!=)==[A-Z])',
            r'==French==\n(.*\n)*?(----)'
        )
    elif (
            args.language_mutation_of_wikti == 'en' and
            args.language_to_extract == 'ru'
    ):
        language_content = (
            r'==Russian==\n(.*\n)*.*',
            r'((?<!=)==[A-Z])',
            r'==Russian==\n(.*\n)*?(----)'
        )
        if args.category == 'diminutives':
            patterns = (
                r'iminutive form of {\{(.*?)\}',
                r'iminutive of {\{(.*?)\}',
                r'diminutive=(.*?)[\|\}]',
                r'[Aa] \[*little()',
                r'[Aa] \[*low()'
                r'[Aa] \[*small()'
            )

    elif (
        args.language_mutation_of_wikti == 'ru' and
        args.language_to_extract == 'ru'
    ):
        language_content = (
            r'==Русский==\n(.*\n)*.*',
            r'((?<!=)==[А-Я])',
            r'==Русский==\n(.*\n)*?(----)'
        )

        if args.category == 'diminutives':
            patterns = (
                r'[Уу]меньш.* {\{(.*?)\}',
            )
        else:
            logging.error('Unknown combination of semantic category settings.')
            sys.exit()

    else:  # Incorrect combination
        logging.error('Unknown combination of language settings.')
        sys.exit()

    # obtain relevant entries for a given language
    relevant_entries = load_relevant_entries(
        path=args.input_data, language_content=language_content
    )

    # extract information from entries
    extracted_data = extract_information(
        data=relevant_entries, patterns=patterns
    )

    # clean derivational parent(s)
    cleaning_function = '_'.join(
        (
            args.language_mutation_of_wikti,
            args.language_to_extract,
            'clean_parent_strings'
        )
    )
    resulting_data = eval(cleaning_function)(extracted_data)

    return list(set(resulting_data))


# run this script
if __name__ == '__main__':
    # parse initial arguments and extract data
    args = parser.parse_args()
    results = main(args)

    # save extracted data
    with open(args.output_data, mode='w', encoding='U8') as f:
        for item in results:
            print(*item, sep='\t', file=f)
