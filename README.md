# 🗺️ DMAP: A Distribution Map for Text

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DMAP (ICLR 2026) is a mathematically grounded method that maps a text, via a language model, to a set of samples in the unit interval that jointly encode rank and probability information. This representation enables efficient, model-agnostic analysis and supports a range of applications.

DMAP works effectively with small evaluator language models such as [OPT-125m](https://huggingface.co/facebook/opt-125m) that easily run on consumer hardware. 

## ✨ Key Features

- 🎯 **Intuitive Visualization**: transform text into simple, informative, representations for downstream analysis 
- 🔧 **Easy Integration**: Simple API that works with popular NLP libraries (transformers, scikit-learn, etc.)
- 📊 **Rich Analytics**: Built-in tools for quantitative and qualitative analysis of distribution patterns
- 🎨 **Customizable**: Easily plug-in new visualisations or analysis methods
- 📖 **Interactive demo**: Get up and running with DMAP in a few minutes

## 🚀 Quick Start

To install, simply run:
```bash
pip install git+https://github.com/Featurespace/dmap.git
```

Then, you may use DMAP as follows.

```python
from dmap import DMAP

# Create and fit DMAP.
dmap = DMAP(evaluator_model='facebook/opt-125m')
text_map = dmap.fit(["The robot was dancing in the rain"])

# Visualize your DMAP samples.
dmap.plot()
```

For a more detailed example, we recommend cloning the repository and playing with our [interactive demo](./demo.ipynb).

## 📄 Citation

If you use DMAP in your research, please cite our paper accepted at ICLR 2026:

```bibtex
@article{dmap2025,
  title={DMAP: A Distribution Map for Text},
  author={Tom Kempton, Julia Rozanova, Parameswaran Kamalaruban, Maeve Madigan, Karolina Wresilo, Yoann Launay, David Sutton, and Stuart Burrell},
  year={2026},
  url={https://openreview.net/forum?id=SPElkPRurl}
}
```
