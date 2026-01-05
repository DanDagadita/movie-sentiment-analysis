#import "@preview/grape-suite:3.1.0": seminar-paper
#import seminar-paper: definition, sidenote

#let definition = definition.with(figured: true)

#show: seminar-paper.project.with(
  title: [
    #v(-60cm)
    #text("Sentiment Analysis of Movie Reviews")
  ],
  subtitle: "Classifying movie reviews as positive or negative",

  university: [University of Craiova],
  faculty: [Faculty of Automation, Computers and Electronics],
  institute: none,
  docent: [Prof. dr. ing. Costin Bădică],
  seminar: none,
  semester: none,
  date: datetime.today().display("[month repr:long] [year]"),

  author: "Dan-Cosmin Dăgădiță",
  email: "dagadita.dan.v7d@student.ucv.ro",
  address: none,
  show-declaration-of-independent-work: false,
)

= Introduction

Sentiment analysis is the science of teaching computers how to understand human emotions within text. Instead of a human having to read thousands of movie reviews to see if a film was well-received, we use Natural Language Processing to automatically label them as "Positive" or "Negative." This is useful for businesses and researchers because it allows them to process unstructured data-like social media comments or customer feedback @d2l_rnn.

#definition[
  *Natural Language Processing (NLP):* A field of Artificial Intelligence that focuses on the interaction between computers and human language. Its goal is to enable computers to understand, interpret, and generate text in a way that is valuable @d2l_rnn.
]<def-nlp>

However, understanding human language is a difficult task for a machine. The challenge is in mostly these two areas:
- *Language Nuances:* Humans often use sarcasm, complex descriptions, or words that change meaning depending on the context.
- *Long-Term Dependencies:* In a long movie review, a critic might start with a specific detail and not reveal their final opinion until the very end. A computer needs to "remember" the beginning of the story to understand the conclusion.

To solve these problems, this report uses *Deep Learning*, specifically *Recurrent Neural Networks (RNNs)*. Unlike traditional programs that look at words in isolation, an RNN processes text as a sequence. It reads one word at a time while maintaining an internal "memory" of what it has already seen @d2l_rnn.

In this project, we specifically focus on *Long Short-Term Memory (LSTM)* and *Gated Recurrent Units (GRU)*. These are "smarter" versions of the standard RNN. They were created to fix a specific technical flaw called the *vanishing gradient*, which often causes basic models to "zone out" or lose their way during long sentences @d2l_modern.

#definition[
  *Vanishing Gradient:* Think of this as a "fading signal." When a model is trying to learn from a very long sentence, the signal it uses to update its weights gets weaker and weaker as it travels back to the start of the sentence. Eventually, the signal disappears (vanishes), and the model stops learning @d2l_modern.
]<vanishing-gradient>

= Deep Learning Methodology

The methodology used in this project follows a *sequence-to-vector* architecture @d2l_rnn. The process is broken down into four main steps:

- *Text Vectorization:* This is the first step where the computer turns sentences into a list of ID numbers. Since computers don't understand letters, we give the 10,000 most common words their own "ID number" @d2l_rnn.

- *Word Embeddings:* After we have the ID numbers, we turn them into *dense vectors*. Think of this as giving every word its own set of coordinates on a map of "meaning" @d2l_rnn. Pre-trained word embeddings for this project were sourced from Stanford @stanford_glove and Kaggle @kaggle_glove.

#definition[
  *Word Embeddings:* A technique where words are represented as vectors of numbers. This allows the computer to calculate the "distance" between words. For example, "king" and "queen" would be mathematically closer to each other than "king" and "bicycle" @d2l_rnn.
]<def-embeddings>

- *Recurrent Layers:* This is the "brain" of the model that reads the review word by word @d2l_rnn.
  - *SimpleRNN:* The most basic version; it remembers the previous word.
  - *LSTM and GRU:* Advanced versions that have a "long-term memory" @d2l_modern.

- *Bidirectionality:* Normally, a model reads from left to right. By making it *Bidirectional*, we let the model read the review in both directions simultaneously to capture full context @d2l_modern.

= Software Design and Implementation

The software for this project was built using *TensorFlow 2* and *Keras*. Instead of writing six separate programs, we created one flexible function called `build_model`. This acted like a *model factory*: we could tell it exactly what kind of "brain" to use and whether to use pretrained GloVe embeddings or start from scratch.

To keep the training process fast and smooth, we used a specialized *Data Pipeline* (`tf.data.Dataset`).
- *Batching:* We grouped data into sets of 128.
- *Prefetching:* A trick where the computer prepares the *next* batch of data while the model is still processing the current one.

One of the biggest risks in AI is *overfitting*. To prevent this, we used *dropout layers*.

#definition[
  *Overfitting:* A mistake where a model learns the training data *too* well, including its random noise and specific examples. This makes the model "stiff"-it performs perfectly on the training data but fails on new, real-world data because it hasn't learned the general pattern.
]<def-overfitting>

#definition[
  *Dropout:* A regularization technique where we randomly "ignore" some neurons during training. This prevents the model from becoming too reliant on specific words and forces it to learn more robust patterns.
]<def-dropout>

*Infrastructure and Reproducibility*

The project is designed for portability using Docker Compose and a Google Colab Jupyter environment. The local runtime setup utilizes a dedicated container to handle the Nvidia GPU environment via the Nvidia Container Toolkit. Data and embeddings are managed in a local `files` directory, ensuring the project remains independent of specific cloud-only features. This allows the notebooks to be executed consistently across local IDEs like VS Code and the hosted Google Colab environment.

= Dataset

For this project, we used the *IMDB Movie Reviews Dataset* @kaggle_imdb. The dataset contains *50,000 reviews*. These are described as "highly polar," meaning they are either clearly positive or clearly negative @d2l_rnn.

To make sure the model actually learns, we split the data into two groups:
- *Training Set (40,000 reviews):* The "textbook" used to learn.
- *Testing Set (10,000 reviews):* The "final exam" with reviews the model has never seen before.

When the model "studies," we use a process called *batching*.

#definition[
  *Batch Size:* The number of training examples processed in one pass. A batch size of 128 means the model looks at 128 reviews, calculates its error, updates its weights, and then moves to the next 128.
]<def-batch>

= Experiments and Results

After training the six different models for 3 rounds (epochs), we compared their performance.

#figure(image("plot.png"))

#definition[
  *Epoch:* One complete pass of the entire training dataset through the neural network @d2l_rnn. Training for multiple epochs allows the model to refine its "guesswork" and become more accurate.
]<def-epoch>

The *standard LSTM* was the top performer with *89.94% accuracy*. Even though it is simpler than the Bidirectional version, it was the most efficient @d2l_modern.

- The *GRU (89.31%)* and *Bidirectional LSTM (89.45%)* were very close behind the winner @d2l_modern.
- The *Simple RNN (83.71%)* had the lowest accuracy because of its "short-term memory" issues @d2l_rnn.

The *GloVe* models were expected to be the strongest but performed the worst (around *80-82%*). This is likely because GloVe was trained on general Wikipedia facts, whereas movie reviews use very specific, emotional language that the "random" models learned better from scratch.

= Conclusion

The experiments show that the *standard LSTM* was the most successful model (89.94%). It performed better than the *Simple RNN* because it can remember important words even in very long reviews @d2l_modern.

A surprising result was that *randomly initialized embeddings* performed better than *pretrained GloVe embeddings*. This is likely because the random embeddings were allowed to learn the specific "slang" of movie critics, whereas GloVe was restricted by its general-purpose training.

Additionally, making the models more complex (Bidirectional) did not help much. This suggests that for a simple "Positive or Negative" task, a standard LSTM is already powerful enough @d2l_modern.

#pagebreak()
#bibliography("report.bib", title: "References")
