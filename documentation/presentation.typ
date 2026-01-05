#import "@preview/grape-suite:3.1.0": slides
#import slides: *

#show: slides.with(
  no: none,
  series: [Sentiment Analysis of Movie Reviews],
  title: [Classifying movie reviews as positive or negative],

  show-semester: false,
  date: "January 2026",

  outline-title-text: "Presentation Overview",

  author: "Dan-Cosmin Dăgădiță",
  email: link("mailto:dagadita.dan.v7d@student.ucv.ro"),
)

#slide[
  = Introduction & Motivation

  *Why sentiment analysis?*
  - Massive amounts of unstructured data (social media, reviews).
  - Automation of opinion categorization for business insights.

  *Inherent challenges:*
  - *Nuances:* Sarcasm, irony, and context-dependent meanings.
  - *Long-term dependencies:* Sentiment often depends on words far apart in a text.

  *Goal:* Compare different deep learning architectures to find the most efficient solution for movie review classification.
]

#slide[
  = Deep Learning Methodology

  == The sequence-to-vector approach
  - Text is processed as a varying-length sequence of words.
  - The network compresses the sequence into a single probability score.

  == RNN architectures tested:
  1. *Simple RNN:* The baseline recurrent model.
  2. *LSTM (Long Short-Term Memory):* Gated memory cells.
  3. *GRU (Gated Recurrent Unit):* A lighter, simplified LSTM.
  4. *Bidirectional:* Processing text forward and backward simultaneously.
]

#slide[
  = The Vanishing Gradient Problem

  #definition[
    A phenomenon where signals (gradients) used to update network weights decay exponentially as they travel back through long sequences.
  ]

  - Standard RNNs "forget" the beginning of long reviews.
  - *LSTM/GRU solution:* They use *gates* to regulate information flow, effectively maintaining a "memory" of important sentiment-carrying words over long distances.
]

#slide[
  = Dataset & Preprocessing

  == The IMDB dataset @kaggle_imdb
  - *Size:* 50,000 highly polar reviews.
  - *Split:* 80% Training (40,000) / 20% Testing (10,000).
  - *Labels:* Binary (Positive = 1, Negative = 0).

  == Preprocessing pipeline:
  - *Vectorization:* Mapping top 10,000 words to unique IDs @d2l_rnn.
  - *Padding:* Making all reviews in a batch (size 128) equal length.
  - *Masking:* Ensuring the RNN ignores the "0" padding tokens.
]

#slide[
  = Word Embeddings: Random vs GloVe

  - *Random initialization (128d):* Weights learned from scratch on the IMDB data.
  - *Pre-trained GloVe (100d):* Learned from billions of words on Wikipedia @stanford_glove.

  #hint[
    *The hypothesis:* We expected GloVe to provide a superior semantic foundation for the model to understand word relationships out of the box.
  ]
]

#slide[
  = Experimental Results

  #figure(image("plot.png", width: 60%))

  *Analysis:*
  - *Winner:* Standard LSTM provided the best balance of speed and accuracy.
  - *The surprise:* GloVe performed worse. Task-specific random embeddings captured the critics' language better than general-purpose GloVe vectors.
]

#slide[
  = Technical Implementation

  == Reproducibility & Environment
  - *Framework:* TensorFlow 2 / Keras API.
  - *Infrastructure:* Dockerized environment with Nvidia GPU support.
  - *Architecture:* Modular `build_model` factory for fair comparison.

  #task[
    All code and local runtime setup details are available on GitHub:
    #link("https://github.com/DanDagadita/movie-sentiment-analysis")
  ]
]

#slide[
  = Conclusion

  - Gated architectures (LSTM/GRU) are essential for long-text sentiment analysis.
  - Local, domain-specific training can outperform general pre-trained embeddings.
  - Bidirectionality adds complexity but provided marginal gains for this specific task.
]

#slide[
  #show: align.with(center + horizon)
  #heading(outlined: false)[Thank you for your attention!]
]

#pagebreak()
#bibliography("report.bib", title: "References")
