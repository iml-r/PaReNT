```
  ███████          ███████          ██      █  ███████████
  █     █          █     █          ██      █       █
  █     █          █     █          █ ██    █       █
  █    █           █    █           █   █   █       █
  ██████           ██████           █   █   █       █
  █       ████     █  █      ███    █    ██ █       █
  █           █    █   █    █   █   █     █ █       █
  █       █████    █    █   █████   █      ██       █
  █      █    █    █    █   █       █      ██       █
  █       ████ █   █     █   ████   █       █       █
 ┏━━┓     ┏━━┓      ┏━━┓     ┏━━┓      ┏━━┓       ┏━━┓
 ┃  ┠┐    ┃  ┠┐     ┃  ┠─┐  ┌┨  ┃ ┌────┨  ┃      ┌┨  ┃
 ┗┯┯┛│    ┗┯┯┛│     ┗┯┯┛ │  │┗┯┯┛ │    ┗┯┯┛      │┗┯┯┛
  ││ │     ││ │  ┌───┘│╔═╪══╩═╪╪══╪═════╩┼───────┘ ││
  ││ └─────┼┼─╧══╪════╩╬═╪════╬╩══╪══╗  ┌┘         ││
  │└───────┼╧════╪═════╬╗│   ╔╝   ╠──╫──┼──────────┘│
  │        │    ┏┷━┓   ║║│ ┏━┷┓   ║  ║┏━┷┓          │
  └────────╧════┨  ┠═══╝╚╧═┨  ┠═══╝  ╚┨  ┠──────────┘
                ┗┯┯┛       ┗┯┯┛       ┗┯┯┛
         ┌───────┘│╔════════╧┼─────────┤└───────┐
         │        └╫───┬─────┼─────────┼─────┐  │
         │  ╔══════╝   │  ┌──┴────┐    │     │  │
         │  ║          │  ╠───────┼──┬─┘     │  │
         │  ║          │  ║       │  │       │  │
         └┏┓╝          └┏┓╝       └┏┓┘       └┏┓┘
          ┗┛            ┗┛         ┗┛         ┗┛
        Parent       Retrieval   Neural      Tool
```

## Functionality

PaReNT is a free multilingual tool performing parent retrieval and word formation classification. 
It is primarily intended for researchers in word formation and morphology, but can be found useful by anyone in computational linguistics or natural language processing.

### Parent retrieval
Parent retrieval refers to producing the ancestor(s) of a given input word. For instance, the ancestor (or parent) of the English word _development_ is _develop_; and the parents of _waterfall_ are _water_ and _fall_. It is an extension of lemmatization, an evolution of stemming, and a generalization of compound splitting. If you find any of these tasks useful, there is a good chance you might find parent retrieval useful as well.

### Word formation classification
Very broadly speaking, words can be separated into three categories: 

 0. unmotivated words such as _dog_, which have no parent*,
 1. derivative words such as the _development_, which have one, and
 2. compound words such as _waterfall_, which have two or more. 

PaReNT has a specialized classification module which classifies each word into one of the three categories independently of the parent retrieval process.

*For technical reasons, we model unmotivated words as being their own parents.

## Performance
The table shown here describes the performance of the model used in the current version of this GitHub repo. 
It may differ from the performance of the model described in the associated LREC paper.


| Language | Retrieval accuracy | Classification accuracy | Balanced classification accuracy |
|----------|--------------------|-------------------------|----------------------------------|
|    cs    |        0.69        | 0.96                    |               0.66               |
|    de    |        0.62        | 0.95                    |               0.84               |
|    en    |        0.75        | 0.88                    |               0.87               |
|    es    |        0.83        | 0.98                    |               0.66               |
|    fr    |        0.52        | 0.93                    |               0.49               |
|    nl    |        0.58        | 0.91                    |               0.83               |
|    ru    |        0.68        | 0.98                    |               0.82               |
|  Total   |        0.67        | 0.94                    |               0.74               |

## Application
As far as we know, the tool has been used in the following papers:

- [Identification and analysis of Czech equivalents of German compounds](https://dspace.cuni.cz/bitstream/handle/20.500.11956/127960/130309002.pdf?sequence=1&isAllowed=y)

## Usage
There are three ways to use PaReNT:

- interactive toy mode;
- CLI tool;
- Python API.

We will go through their usage one by one, but first, the tool has to be installed.

### Installation
Installation consists of downloading the source files and installing the required Python packages.

```bash
git clone https://github.com/iml-r/PaReNT/
cd PaReNT
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Interactive mode
Simply run the PaReNT.py file in your console with the -i flag.

```bash
python3 PaReNT.py -i
```

Then, an animation showcasing the logo will play, while the required packages (primarily TensorFlow) are being loaded, which takes a while. 
Depending on your system, the animation may be shorter than the loading time, so please wait for the "Word:" prompt to appear.
Once it does, you can input a word of your choosing, followed by an [ISO 639-1 language code](https://www.loc.gov/standards/iso639-2/php/code_list.php).

### CLI tool
Specific format .tsv files can be pipelined into PaReNT. There has to be at least one column called **lemma**, which contains lemmatized words from one of the languages supported by PaReNT (see Section **Performance**).
Additionally, the .tsv file should contain a **language** column as well, containing langauge codes. PaReNT will run even if the **language** column is absent (the neural network will try to infer the language), but this feature is untested and not recommended unless strictly necessary.

Once you have such a file, e.g. _test.tsv_ (example file contained in this very repo), simply do this:

```bash
python3 PaReNT.py < test.tsv > test_output.tsv
```

If there are more columns in your input .tsv file, PaReNT will keep them unchanged. It will add the following 6 columns:

1) PaReNT_retrieval_best:                Best parent(s), selected from PaReNT_retrieval_candidates based on columns 4), 5) and 6).
2) PaReNT_retrieval_greedy:              Parent(s) retrieved using greedy decoding.
3) PaReNT_retrieval_candidates:          All candidates retrieved using beam search decoding, sorted by score.
4) PaReNT_Compound_probability:          Estimated probability the word is a Compound.
5) PaReNT_Derivative_probability:        Estimated probability the word is a Derivative.
6) PaReNT_Unmotivated_probability        Estimated probability the word is Unmotivated

### Python API
Simply clone the repo into you project's directory and install the required packages into your project's virtual environment. We cannot guarantee performance listed in **Performance** unless the specific versions listed in _requirements.txt_ are respected.

Once that's done, initialize the model:

```python
import PaReNT.PaReNT_core as PaReNT

##Specify one of the models in the _model_ directory -- currently, only one is available
model = PaReNT.PaReNT("e13-arc=FINAL5-clu=True-bat=64-epo=1000-uni=2048-att=512-cha=64-tes=0-tra=1-len=0.0-fra=1-lr=0.0001-opt=Adam-dro=0.2-rec=0.5-l=l1-use=1-neu=0-neu=0-sem=0")
```
There are two recommended methods to use -- _.classify()_ and _.retrieve_and_classify()_. 

```python
# Both methods expect a list of tuples: [(<language_code>, <lexeme>)]
lexemes = [("en", "development"), ("de", "Hochschule")]

classification = model.classify(lexemes)

#The retrieve_and_classify method uses beam search decoding. This allows you
#to utilize the try_candidates and threshold options. Try_candidates is an integer
#telling PaReNT to return the n best candidate parent sequences it can guess,
#and threshold tells it how many.  
retrieval_and_classification = model.retrieve_and_classify(lexemes, try_candidates=True, threshold=10)
```
