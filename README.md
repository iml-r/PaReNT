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
Parent retrieval refers to producing the ancestor(s) of a given input word. For instance, ancestor (or parent) of the English word _development_ is _develop_; and the parents of _waterfall_ are _water_ and _fall_. It is an extension of lemmatization, an evolution of stemming, and a generalization of compound splitting. If you find any of these tasks useful, there is a good chance you might find parent retrieval useful as well.

### Word formation classification
Very broadly speaking, words can be separated into three categories: 

 0. unmotivated words such as _dog_, which have no parent*,
 1. derivative words such as the _development_, which have one, and
 2. compound words such as _waterfall_, which have two or more. 

PaReNT has a specialized classification module which classifies each word into one of the three categories independently of the parent retrieval process.

*For technical reasons, we model unmotivated words as being their own parents.

## Performance
The table shown here describes the performance of the model used in the current version of this GitHub repo. 
It may differ from the the performance of the model described in the associated LREC paper.

## Application
As far as we know, the tool has been used in the following papers:

- [Identification and analysis of Czech equivalents of German compounds](https://dspace.cuni.cz/bitstream/handle/20.500.11956/127960/130309002.pdf?sequence=1&isAllowed=y)

## Usage
There are three ways to use PaReNT:

- interactive toy mode;
- CLI tool;
- Python package.

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
Once it does, you can input a word of your choosing, followed by a language token (refer to table )



