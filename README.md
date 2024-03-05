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

PaReNT is a free multilingual tool performing parent retrieval and word formation classification. 
It is primarily intended for researchers in word formation and morphology, but can be found useful by anyone in computational linguistics or natural language processing.

### Parent retrieval
Parent retrieval refers to producing the ancestor(s) of a given input word. For instance, ancestor (or parent) of the English word _development_ is _develop_; and the parents of _waterfall_ are _water_ and _fall_. It is an extension of lemmatization, an evolution of stemming, and a generalization of compound splitting. If you find any of these tasks useful, there is a good chance you might find parent retrieval useful as well.

### Word formation classification
Very broadly speaking, words can be separated into three categories -- unmotivated words such as _dog_, which have no parent*

* For the purposes of PaReNT, this is modelled as 
## Functionality

### Installation
