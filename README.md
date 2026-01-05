# Sentiment Analysis on IMDB Movie Reviews

## Google Colab links

Final project: https://colab.research.google.com/drive/1ARohxdSlNSFj7VJmpFBULC8kmgm4V6WJ?usp=sharing

TensorFlow tutorial project: https://colab.research.google.com/drive/1xRWLs52TTSiDeP_h_ONdIzV8EfUxtcPx?usp=sharing

## Project Overview

This project implements and compares multiple Deep Learning architectures for sentiment analysis on the IMDB movie reviews dataset. The study investigates the effectiveness of different RNN variants (SimpleRNN, LSTM, GRU) with and without pre-trained GloVe embeddings.

## Prerequisites

This project requires a machine with an Nvidia GPU and its drivers, Docker Compose, and the Nvidia Container Toolkit. It has been tested only on an Arch Linux machine, for more information about Google Colab local runtimes, click [here](https://research.google.com/colaboratory/local-runtimes.html).

## Starting

**1.** In a terminal, clone this repository: `git clone https://github.com/DanDagadita/movie-sentiment-analysis.git`

**2.** You need to download the dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), which is inside an archive which you need to extract it from and place it into the `files` directory.

The GloVe word embeddings file also needs to be downloaded, either from [Kaggle](https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt) or from [Stanford](https://nlp.stanford.edu/data/glove.6B.zip). Only the `glove.6B.100d.txt` file needs to be extracted from either archive, then placed in the `files` directory. The directory structure should now look like this:

```
├── documentation
│   ├── plot.png
│   ├── presentation.pdf
│   ├── presentation.pptx
│   ├── presentation.typ
│   ├── report.bib
│   ├── report.pdf
│   └── report.typ
├── exported
│   ├── final.py
│   └── tutorial.py
├── files
│   ├── .gitignore
│   ├── glove.6B.100d.txt
│   └── IMDB Dataset.csv
├── notebooks
│   ├── final.ipynb
│   └── tutorial.ipynb
├── compose.yml
├── jupyter_config.py
├── LICENSE
├── NOTICE
└── README.md
```

**3.** Enter the cloned repository's folder, then run `docker compose up -d`, which starts the container containing the Google Colab Jupyter environment.

**4.** Upon opening this project in your text editor/IDE of choice, and opening the `.ipynb` files from the `notebooks` folder, you have the choice to connect to the local runtime using `http://127.0.0.1:9000`. I used VSCode, using the suggested Jupyter and Python extensions.

**IMPORTANT NOTE:** When connecting Google Colab to this local runtime, the functions from `google.colab` libraries will not work. If those functions are desired, use Google's hosted runtime. The `.ipynb` files in this project use local files and no Google Colab features. But links to native Google Colab features exist at the top of this document, but rely on the user running the notebooks to have the content in `files` folder in their Google Drive account, requiring manual intervention.
